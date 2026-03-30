import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_aggregation_metadata(res_dir):
    """
    Reads trainer_config.json to find observed_root and load the S matrix.
    """
    config_path = Path(res_dir) / "trainer_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    observed_root = Path(config.get("observed_root", ""))
    s_path = observed_root / "S.pkl"
    if not s_path.exists():
        raise FileNotFoundError(f"S.pkl not found at {s_path}")

    with open(s_path, "rb") as f:
        S = np.load(f, allow_pickle=True)

    return S


def get_dynamic_horizons(S):
    """
    Given an aggregation matrix S, returns a list of the number of nodes
    at each hierarchical level, ordered top-down.
    """
    S_vals = np.array(S)

    # Count how many base variables aggregate to each node (non-zero elements per row)
    non_zeros_per_row = np.count_nonzero(S_vals, axis=1)

    # Nodes with the same number of non-zero elements are at the same level
    unique_levels, counts = np.unique(non_zeros_per_row, return_counts=True)

    # Order by unique_levels descending (most aggregated -> least aggregated)
    # This corresponds to top-down hierarchical ordering.
    sort_idx = np.argsort(unique_levels)[::-1]
    node_counts = counts[sort_idx].tolist()

    return node_counts


def to_plot_dynamic(data, node_counts):
    """
    Chunks a 1D loss array dynamically based on the inferred node counts per level.
    Upsamples each level to have the same length as the base level frequency.
    Returns a dictionary mapping 'aggregation_level' -> 'upsampled_array'.
    """
    res = {}
    current_idx = 0
    total_base_nodes = max(node_counts)

    for count in node_counts:
        agg_level = total_base_nodes // count
        res[agg_level] = []

        # Each data point is repeated 'agg_level' times to match the base length
        for _ in range(count):
            val = data[current_idx]
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                val = val[0]

            res[agg_level].extend([val] * agg_level)
            current_idx += 1

    return res


def plot_hierarchy(model_mapping, title="Reconciliation Result", display_freqs=None):
    """
    A unified function to plot multiple models at once.

    Args:
        model_mapping (dict): A dictionary mapping a method name to its result directory path
                                or a direct pickle file path.
                                Example: {'Base': '/path/to/base.pkl', 'Angular': '/path/to/dir'}
        title (str): Title for the figure.
        display_freqs (list or int): List of aggregation levels to plot, or max number of subplots to show.
    """
    res_plot = {}
    overall_loss = {}

    # 1. Infer dynamic horizons from the first valid directory
    node_counts = None
    for method, path_str in model_mapping.items():
        path = Path(path_str)
        if path.is_dir():
            try:
                S = get_aggregation_metadata(path)
                node_counts = get_dynamic_horizons(S)
                break
            except Exception as e:
                print(f"Warning: Could not infer from {path}: {e}")

    if node_counts is None:
        print(
            "Warning: Could not extract hierarchy from directories, using default temporal node counts."
        )
        node_counts = [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]

    # 2. Load data for each model
    for method, path_str in model_mapping.items():
        path = Path(path_str)
        loss_file = path / "test_loss.pkl" if path.is_dir() else path

        if not loss_file.exists():
            print(f"Skipping {method}: {loss_file} does not exist.")
            continue

        with open(loss_file, "rb") as f:
            loss = np.load(f, allow_pickle=True)

        res_plot[method] = to_plot_dynamic(loss, node_counts)
        overall_loss[method] = loss.sum()

    if not res_plot:
        print("No valid data found to plot.")
        return

    overall_best_method_nm = min(overall_loss, key=lambda k: overall_loss[k])

    # 3. Determine which frequencies to display
    available_freqs = list(list(res_plot.values())[0].keys())

    if display_freqs is None:
        freqs_to_plot = available_freqs[:6]
    elif isinstance(display_freqs, int):
        freqs_to_plot = available_freqs[:display_freqs]
    else:
        freqs_to_plot = [f for f in display_freqs if f in available_freqs]

    n_plots = len(freqs_to_plot)
    if n_plots == 0:
        print("No frequencies matched availability.")
        return

    cols = min(3, n_plots)
    rows = int(np.ceil(n_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(
        f"{title} (Best: {overall_best_method_nm})",
        fontsize=23,
        fontweight="bold",
        y=0.98,
    )

    if n_plots == 1:
        axes_flat = [axes]
    elif rows == 1 or cols == 1:
        axes_flat = axes
    else:
        axes_flat = axes.flatten()

    for idx, horizon in enumerate(freqs_to_plot):
        ax = axes_flat[idx]

        horizon_scores = {}
        for method in res_plot:
            ax.plot(res_plot[method][horizon], label=method)
            horizon_scores[method] = np.sum(res_plot[method][horizon])

        horizon_best_method_nm = min(horizon_scores, key=lambda k: horizon_scores[k])

        # Calculate improvement against base if present
        base_name = None
        for candidate in ["base", "Base"]:
            if candidate in res_plot:
                base_name = candidate
                break

        if base_name and horizon_best_method_nm != base_name:
            base_score = horizon_scores[base_name]
            best_score = horizon_scores[horizon_best_method_nm]
            improvement = ((base_score - best_score) / base_score) * 100
            sub_title = f"Aggregation level: {horizon}\nBest: {horizon_best_method_nm}, Impr: {improvement:.2f}%"
        else:
            sub_title = f"Aggregation level: {horizon}\nBest: {horizon_best_method_nm}"

        ax.set_title(sub_title)
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_plots, rows * cols):
        axes_flat[idx].set_visible(False)

    plt.subplots_adjust(wspace=0.35, hspace=0.25, top=0.89)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
