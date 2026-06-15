import pandas as pd
import numpy as np




def clean_and_reconcile_pricing(df: pd.DataFrame, cols: list[str]):# -> pd.DataFrame:

    LP = cols[0]
    DP = cols[1] 
    DC = cols[2]

    #print(LP, DP, DC)
    #print(cols)
    #return
    df_copy = df.copy()


    # 0) Ensure numeric
    df_copy[cols] = df_copy[cols].apply(pd.to_numeric, errors='coerce')

    # 1) If any of the three is NaN, set all three to 0.0 (your original rule)
    mask_nan = df_copy[cols].isna().any(axis=1)
    df_copy.loc[mask_nan, cols] = 0.0

    # 2) Fix inversions: list < discounted  -> set discounted = list, discount% = 0.0
    mask_inverted = df_copy[LP] < df_copy[DP]
    # set discounted to list for those rows
    df_copy.loc[mask_inverted, DP] = df_copy.loc[mask_inverted, LP].to_numpy()
    # set discount_percentage to 0.0 for those rows
    df_copy.loc[mask_inverted, DC] = 0.0

    # 3) Recompute discount_percentage from prices for ALL rows
    #    Guard against division by zero or negative list prices.
    lp = df_copy[LP]
    dp = df_copy[DP]

    with np.errstate(divide='ignore', invalid='ignore'):
        recomputed = (lp - dp) / lp

    # Replace inf/-inf with NaN, then fill NaN where appropriate
    recomputed = recomputed.replace([np.inf, -np.inf], np.nan)

    # If list price <= 0 or recomputed is NaN, set discount to 0.0 (adjust to your policy)
    invalid_list = lp <= 0
    recomputed = recomputed.where(~invalid_list, 0.0).fillna(0.0)

    # (Optional) Clip to [0, 1] so no values exceed 100% or go negative after fixes
    recomputed = recomputed.clip(lower=0.0, upper=1.0)

    # Write back with .loc to avoid chained assignment warning
    df_copy.loc[:, DC] = recomputed

    # 4) (Optional) Quick audit
    n_inverted = int(mask_inverted.sum())
    n_nan_rows = int(mask_nan.sum())
    print(f"Rows set to 0 due to NaNs in any of {cols}: {n_nan_rows}")
    print(f"Rows corrected where {LP} < {DP}: {n_inverted}")
    print(df_copy[DC].describe(percentiles=[0.01, 0.5, 0.99]))
    return df_copy



def add_seasonal_harmonics(df, timestamp_col, periods):
    """
    Adds seasonal harmonic (sine/cosine) features to a dataframe based on a POSIX timestamp.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe.
    timestamp_col : str
        The name of the column containing POSIX timestamps (in seconds).
    periods : tuple of tuples
        A tuple containing tuples of (period_name, period_in_seconds).
        Example: (('weekly', 604800), ('yearly', 31536000))
        
    Returns:
    --------
    pd.DataFrame
        A new dataframe with the added harmonic columns.
    list
        A list of strings containing the names of the newly generated columns.
    """
    # Create a copy to avoid SettingWithCopyWarning on the original dataframe
    df_out = df.copy()
    new_cols = []
    
    t = df_out[timestamp_col]
    
    for name, period_seconds in periods:
        sin_col = f'{timestamp_col}_sin_{name}'
        cos_col = f'{timestamp_col}_cos_{name}'
        
        df_out[sin_col] = np.sin(2 * np.pi * t / period_seconds)
        df_out[cos_col] = np.cos(2 * np.pi * t / period_seconds)
        
        new_cols.extend([sin_col, cos_col])
        
    return df_out, new_cols


def filter_rare_categories(df, categorical_cols, threshold, replacement="__NONE__"):
    """
    Replaces rare categories in the specified columns with a replacement value
    rather than dropping the rows.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - categorical_cols (list): List of column names
    - threshold (float): Minimum frequency threshold (0.0 to 1.0)
    - replacement (str): The value to use for rare categories

    Returns:
    - pd.DataFrame: Modified DataFrame
    """
    df_filtered = df.copy()

    for col in categorical_cols:
        # Calculate frequencies
        freq = df_filtered[col].value_counts(normalize=True)
        # Identify categories that meet the threshold
        valid_categories = freq[freq >= threshold].index
        
        # If the column is a Categorical type, we must add the replacement to the categories first
        if pd.api.types.is_categorical_dtype(df_filtered[col]):
            if replacement not in df_filtered[col].cat.categories:
                df_filtered[col] = df_filtered[col].cat.add_categories([replacement])
        
        # Replace values not in valid_categories with the replacement string
        df_filtered.loc[~df_filtered[col].isin(valid_categories), col] = replacement
        
        print(f"Column '{col}': Kept {len(valid_categories)} categories. Others mapped to {replacement}.")

    return df_filtered

def filter_rare_categories_(df, categorical_cols, threshold):
    """
    Removes rows from the DataFrame where any specified categorical column
    has a category whose relative frequency is below the threshold.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - categorical_cols (list): List of column names (categorical)
    - threshold (float): Minimum percentage threshold (0.0 to 1.0)

    Returns:
    - pd.DataFrame: Filtered DataFrame
    """
    df_filtered = df.copy()

    for col in categorical_cols:
        freq = df_filtered[col].value_counts(normalize=True)
        valid_categories = freq[freq >= threshold].index
        #print(valid_categories)



        # Filter rows where the category is common enough
        df_filtered = df_filtered[df_filtered[col].isin(valid_categories)]
        df.reset_index(drop=True, inplace=True)

    return df_filtered

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