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



def filter_rare_categories(df, categorical_cols, threshold):
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

#=============
#USAGE
#=============

# 1. Setup a dummy dataframe with a POSIX timestamp
# (e.g., hourly data starting from Jan 1, 2024)
#timestamps = pd.date_range("2024-01-01", periods=100, freq="h").astype('int64') // 10**9
#df = pd.DataFrame({
#    'posixtimestamp': timestamps,
#    'treatment': np.random.binomial(1, 0.5, 100),
#    'target': np.random.normal(0, 1, 100)
#})

# 2. Define your frequencies in seconds
# Daily: 24 * 60 * 60 = 86,400
# Weekly: 7 * 86,400 = 604,800
# Monthly (approx 30.44 days): 30.44 * 86,400 = 2,630,016
#harmonic_periods = (
#    ('daily', 86400),
#    ('weekly', 604800),
#    ('monthly', 2630016)
#)

# 3. Apply the function
#df_enriched, time_confounder_cols = add_seasonal_harmonics(
#    df=df, 
#    timestamp_col='posixtimestamp', 
#    periods=harmonic_periods
#)

# View the generated column names
#print("New Harmonic Columns:", time_confounder_cols)
# Output: ['posixtimestamp_sin_daily', 'posixtimestamp_cos_daily', ...]

# 4. Use in CausalForestDML
#from econml.dml import CausalForestDML

# Assuming you have other features (X) for heterogeneity
#X = df_enriched[['some_other_feature']] if 'some_other_feature' in df_enriched.columns else None

# Pass the harmonic features directly as confounders (W)
#W = df_enriched[time_confounder_cols]

# Initialize and fit the causal model
#est = CausalForestDML(discrete_treatment=True)

# est.fit(Y=df_enriched['target'], T=df_enriched['treatment'], X=X, W=W)

