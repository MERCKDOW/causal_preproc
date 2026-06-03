import pandas as pd
import numpy as np

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

