import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_average_performance(model_mapping, title="Comparison of Average Loss"):
    """
    Plots the average performance (overall loss) of various models, ranking them.
    Highlights methods containing the word 'angular' in red.

    Args:
        model_mapping (dict): A dictionary mapping a method name to its result directory path or a direct pickle file path.
            Example: {'Base': '/path/to/base.pkl', 'Angular': '/path/to/dir'}
        title (str): Title for the figure.
    """
    overall_loss = {}

    for method, path_str in model_mapping.items():
        path = Path(path_str)
        loss_file = path / "test_loss.pkl" if path.is_dir() else path

        if not loss_file.exists():
            print(f"Skipping {method}: {loss_file} does not exist.")
            continue

        with open(loss_file, "rb") as f:
            loss = np.load(f, allow_pickle=True)
    
        overall_loss[method] = loss.mean()

    if not overall_loss:
        print("No valid data found to plot.")
        return

    # Sort data by loss (smaller is better usually)
    data = sorted(overall_loss.items(), key=lambda x: x[1])
    methods, avg_losses = zip(*data)

    highlight_methods = [m for m in methods if "angular" in m.lower()]

    plt.figure(figsize=(10, 6))

    plt.scatter(avg_losses, methods, s=50)
    ax = plt.gca()
    plt.yticks(range(len(methods)), methods)
    
    for label in ax.get_yticklabels():
        if label.get_text() in highlight_methods:
            label.set_color('red')

    plt.xlabel('Average Loss (CRPS, kw)', fontsize=12)
    plt.ylabel('Methods', fontsize=12)
    
    plt.title(title, fontsize=14)

    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
