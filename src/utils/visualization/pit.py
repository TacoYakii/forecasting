import matplotlib.pyplot as plt 
from matplotlib.figure import Figure
import numpy.typing as npt
import numpy as np 
from typing import Tuple, Optional
from ..metrics.pit import pit_get_values, pit_uniformity_test

def plot_pit_histogram(
    pit_values: npt.NDArray[np.float64],
    bins: int = 10,
    title: str = "PIT Histogram",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show_stats: bool = True
) -> Figure:
    """PIT histogram with statistical tests"""
    # remove NaN
    fig, ax = plt.subplots(figsize=figsize)
    valid_pit = pit_values[~np.isnan(pit_values)]

    ax.hist(
        valid_pit, 
        bins=bins, 
        density=True, 
        alpha=0.7,
        color='skyblue', 
        edgecolor='black'
        ) 

    # Ideal Unifrom Line 
    ax.axhline(
        y=1.0, 
        color='red',
        linestyle='--',
        label='Ideal Uniform Distribution', 
        linewidth=2
        )

    # Statistical test information 
    if show_stats:
        stats = pit_uniformity_test(valid_pit)
        stats_text = f"KS p-value: {stats['p_value']:.3f}\n"
        stats_text += f"Uniform: {'Yes' if stats['is_uniform'] else 'No'}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('PIT Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig

def plot_pit_from_samples(
    forecast_samples: npt.NDArray[np.float64],
    observations: npt.NDArray[np.float64],
    **plot_kwargs
) -> Figure:
    """
    Generate PIT values from forecast samples and plot histogram.
    
    This is a convenience function that combines PIT value calculation and visualization.
    It computes the Probability Integral Transform values from forecast samples and 
    observations, then creates a histogram plot with statistical tests.
    
    Parameters:
        forecast_samples : npt.NDArray[np.float64]
            2D array of shape (n_observations, n_samples) containing forecast samples.
            Each row contains simulation samples for one forecast instance.
        observations : npt.NDArray[np.float64]
            1D array of shape (n_observations,) containing the actual observed values.
        **plot_kwargs : dict
            Additional keyword arguments passed to plot_pit_histogram().
            Common options include:
            - bins (int): Number of histogram bins (default: 10)
            - title (str): Plot title (default: "PIT Histogram")
            - figsize (Tuple[int, int]): Figure size (default: (8, 6))
            - save_path (str): Path to save the plot (default: None)
            - show_stats (bool): Whether to show statistical test results (default: True)
        
    Returns:
        Figure
            Matplotlib Figure object containing the PIT histogram with:
            - Histogram of PIT values
            - Reference line for ideal uniform distribution
            - Statistical test results (if show_stats=True)
            
    Examples:
        >>> import numpy as np
        >>> # Generate synthetic forecast data
        >>> n_obs, n_samples = 500, 1000
        >>> true_values = np.random.normal(0, 1, n_obs)
        >>> observations = true_values + np.random.normal(0, 0.1, n_obs)
        >>> forecast_samples = np.random.normal(true_values[:, None], 1, (n_obs, n_samples))
        >>> 
        >>> # Create PIT histogram directly from samples
        >>> fig = plot_pit_from_samples(
        ...     forecast_samples, 
        ...     observations,
        ...     bins=20,
        ...     title="Wind Power Forecast Calibration",
        ...     save_path="pit_analysis.png"
        ... )
    
    Notes:
        This function is equivalent to:
        ```python
        pit_values = pit_get_values(forecast_samples, observations)
        fig = plot_pit_histogram(pit_values, **plot_kwargs)
        ```
    
    See Also:
        plot_pit_histogram : Plot histogram from pre-computed PIT values
        pit_get_values : Calculate PIT values from forecast samples
        pit_uniformity_test : Statistical test for PIT uniformity
    """

    pit_values = pit_get_values(forecast_samples, observations)
    return plot_pit_histogram(pit_values, **plot_kwargs)