import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_T_histogram(scores, clip_outliers=False):
    """
    Plots a histogram of Treatment values, automatically handling 
    unknown bounds and distributions.
    
    Parameters:
    pdp_scores (array-like): The floating point treatment values.
    clip_outliers (bool): If True, clips data to the 1st and 99th percentiles 
                          to visualize the core distribution if extreme outliers exist.
    """
    # Convert to pandas Series and drop NaNs to prevent plotting errors
    data = pd.Series(scores).dropna()
    
    if len(data) == 0:
        print("Error: The provided data is empty or contains only NaNs.")
        return

    # Calculate summary statistics to display on the plot
    p_min = data.min()
    p_max = data.max()
    p_mean = data.mean()
    p_median = data.median()
    zeros_count = (data == 0.0).sum()
    zeros_pct = (zeros_count / len(data)) * 100

    # Handle extreme outliers for better visualization if requested
    if clip_outliers:
        lower_bound = data.quantile(0.01)
        upper_bound = data.quantile(0.99)
        data = data.clip(lower=lower_bound, upper=upper_bound)
        title_suffix = " (Clipped at 1st & 99th Percentiles)"
    else:
        title_suffix = " (Raw Data)"

    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Plot histogram with KDE (Kernel Density Estimate) and automatic binning
    sns.histplot(data, bins='auto', kde=True, color='steelblue', stat='count')

    # Add vertical lines for mean and median to immediately identify skewness
    plt.axvline(p_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {p_mean:.4f}')
    plt.axvline(p_median, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {p_median:.4f}')

    # Add text box with distribution bounds and zero-inflation check
    stats_text = (
        f"Actual Min: {p_min:.4f}\n"
        f"Actual Max: {p_max:.4f}\n"
        f"Zero Values: {zeros_count} ({zeros_pct:.1f}%)"
    )
    plt.gca().text(0.95, 0.5, stats_text, transform=plt.gca().transAxes, 
                   fontsize=10, verticalalignment='center', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Formatting
    plt.title(f'Distribution of Treatment{title_suffix}', fontsize=14, pad=15)
    plt.xlabel('Treatment', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def plot_cate_heatmaps(
    aggregated_results: Dict[str, Dict[str, Any]], 
    cmap: str = "coolwarm", 
    figsize: tuple = (12, 8)
) -> None:
    """
    Generates partitioned heatmaps for aggregated CATEs per heterogeneous variable.

    Args:
        aggregated_results: Output from `compute_and_aggregate_cates`.
        cmap: Matplotlib colormap string.
        figsize: Tuple defining figure dimensions.
    """
    for cat_col, result in aggregated_results.items():
        data_df = result['data']
        treatment_order = result['treatment_order'] 

        # Calculate indices to draw vertical separator lines between distinct treatment families
        vlines_indices = []
        current_group = treatment_order[0][0]
        
        for idx, (group, _) in enumerate(treatment_order):
            if group != current_group:
                vlines_indices.append(idx)
                current_group = group

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            data_df, 
            annot=True, 
            cmap=cmap, 
            center=0, 
            fmt=".3f",
            cbar_kws={'label': 'Average CATE'}
        )

        # Draw vertical lines to separate treatment families
        for vline_idx in vlines_indices:
            ax.axvline(x=vline_idx, color='black', linewidth=2.5, linestyle='--')

        # Formatting
        plt.title(f'Average Conditional Treatment Effects by {cat_col}', pad=20, fontsize=14)
        plt.xlabel('Treatment Levels (Compared to Baseline)', labelpad=15)
        plt.ylabel(cat_col, labelpad=15)
        
        # Improve X-axis label readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()