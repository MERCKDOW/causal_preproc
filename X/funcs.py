import pandas as pd
import numpy as np

def create_thresholded_heterogeneous_features(
    df: pd.DataFrame,
    categorical_cols: list,
    min_freq_threshold: float = 0.02,
    dtype: type = np.float64
):
    """
    Groups rare categories into an '_NONE_' bucket based on a minimum frequency 
    threshold, then one-hot encodes the remaining top categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe.
    categorical_cols : list
        List of heterogeneous categorical columns to process.
    min_freq_threshold : float, default=0.02
        The relative frequency cutoff (e.g., 0.02 = 2%). Categories below this 
        are grouped into '_NONE_'.
    dtype : type, default=np.float64
        Explicit numerical type casting to prevent downstream type promotion failures.
    """
    df_clean = df.copy()
    encoded_dfs = []
    new_column_names = []
    
    for col in categorical_cols:
        # 1. Calculate relative frequencies of the raw categories
        freqs = df_clean[col].value_counts(normalize=True)
        
        # 2. Identify categories that fall below the threshold
        rare_categories = freqs[freqs < min_freq_threshold].index.tolist()
        
        # 3. Group rare categories into '_NONE_'
        # We also handle cases where the column is a category type
        if isinstance(df_clean[col].dtype, pd.CategoricalDtype):
            if '_NONE_' not in df_clean[col].cat.categories:
                df_clean[col] = df_clean[col].cat.add_categories(['_NONE_'])
                
        df_clean[f'{col}_mapped'] = df_clean[col].replace(rare_categories, '_NONE_')
        
        # 4. One-hot encode the thresholded column
        # We explicitly keep '_NONE_' as the reference group by letting drop_first=False,
        # manually dropping the '_NONE_' column to make it the baseline.
        dummies = pd.get_dummies(
            df_clean[f'{col}_mapped'], 
            prefix=col, 
            dtype=dtype
        )
        
        none_col = f"{col}__NONE_"
        if none_col in dummies.columns:
            # Drop the '_NONE_' column so it becomes the baseline reference group
            dummies = dummies.drop(columns=[none_col])
        else:
            # If no rare categories existed, drop the first category standardly
            dummies = dummies.drop(columns=[dummies.columns[0]])
            
        encoded_dfs.append(dummies)
        new_column_names.extend(dummies.columns.tolist())
        
    # Combine the new features with the original dataframe
    result_df = pd.concat([df] + encoded_dfs, axis=1)
    
    return result_df, new_column_names