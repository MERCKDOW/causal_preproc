import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import KFold
from itertools import combinations
import gc
from category_encoders import CatBoostEncoder
#pip install category_encoders


def multi_ordered_target_encode(
    df: pd.DataFrame, 
    cat_cols: List[str], 
    target_col: str, 
    prior_weight: float = 1.0,
    encode_explicit_combinations: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Performs CatBoost-style ordered target encoding across multiple categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing features and the target.
    cat_cols : List[str]
        A list of string column names to be encoded (minimum 1).
    target_col : str
        The name of the target column (Y) used for the encoding statistics.
    prior_weight : float, default 1.0
        The smoothing parameter (w) to stabilize rare categories.
    encode_explicit_combinations : bool, default False
        If True, creates a joint interaction column combining ALL listed 
        categorical variables and encodes that specific combination space.
        
    Returns:
    --------
    Tuple[pd.DataFrame, List[str]]
        - The updated DataFrame containing the new encoded columns.
        - A list of strings containing the names of the newly generated columns.
    """
    if not cat_cols:
        raise ValueError("The cat_cols list must contain at least one column name.")
        
    # Working copy to avoid mutating the original dataframe in-place
    df_out = df.copy()
    new_column_names = []
    
    # Calculate the global baseline prior
    global_prior = df_out[target_col].mean()
    
    # Optional: Build an explicit joint combination column if requested
    columns_to_process = cat_cols.copy()
    if encode_explicit_combinations and len(cat_cols) > 1:
        combination_col_name = "_X_".join(cat_cols)
        # Concatenate values to create a joint category representation (e.g., "SegmentA_DeviceMobile")
        df_out[combination_col_name] = df_out[cat_cols].astype(str).agg('_'.join, axis=1)
        columns_to_process.append(combination_col_name)

    # Compute vectorized ordered target encoding for each feature
    for col in columns_to_process:
        encoded_col_name = f"{col}_catboost_encoded"
        
        # Calculate cumulative sum of the target for this category up to the current row, 
        # then subtract the current row's target value to ensure strictly out-of-history tracking (1 to i-1)
        cum_sum = df_out.groupby(col)[target_col].cumsum() - df_out[target_col]
        
        # Calculate the cumulative count of appearances prior to the current row (0-indexed)
        cum_count = df_out.groupby(col).cumcount()
        
        # Map back to the dataframe using the smoothed formula
        df_out[encoded_col_name] = (cum_sum + prior_weight * global_prior) / (cum_count + prior_weight)
        new_column_names.append(encoded_col_name)
        
        # Clean up the temporary joint string column if we created one
        if encode_explicit_combinations and col == combination_col_name:
            df_out.drop(columns=[combination_col_name], inplace=True)
            
    return df_out, new_column_names



def catboost_style_encoding(df, categorical_cols, target_col, n_splits=5, global_smoothing=10, max_combination_size=2):
    
    """
    Replicates CatBoost-style target encoding by adding new columns to the input DataFrame.

    Parameters:
    - df: pandas DataFrame (modified in-place)
    - categorical_cols: list of categorical column names
    - target_col: name of the target column
    - n_splits: number of folds for cross-validated encoding
    - global_smoothing: regularization strength
    - max_combination_size: max number of columns to combine

    Returns:
    - df: The modified DataFrame with new encoded columns.
    - new_encoded_cols: list of the newly created column names.
    """

    global_mean = df[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    new_encoded_cols = []

    # Generate all combinations of categorical columns
    all_combinations = []
    for r in range(1, max_combination_size + 1):
        all_combinations += list(combinations(categorical_cols, r))

    for combo in all_combinations:
        # Create a clean feature name
        base_name = "_".join([str(c).replace('_ord', '') for c in combo])
        encoded_col_name = f"target_enc_{base_name}"
        new_encoded_cols.append(encoded_col_name)

        # Create temporary combined key for grouping
        temp_key = "_temp_" + "_".join([str(c) for c in combo])
        df[temp_key] = df[list(combo)].astype(str).agg("_".join, axis=1)

        # Initialize the new column with NaN
        df[encoded_col_name] = np.nan

        for train_idx, val_idx in kf.split(df):
            # Compute stats on the training folds
            stats = df.iloc[train_idx].groupby(temp_key)[target_col].agg(['mean', 'count'])

            # Apply smoothing
            smoothed_values = (
                (stats['mean'] * stats['count'] + global_mean * global_smoothing) /
                (stats['count'] + global_smoothing)
            )

            # Map to the validation fold
            df.loc[df.index[val_idx], encoded_col_name] = df.iloc[val_idx][temp_key].map(smoothed_values)
            gc.collect()

        # Fill missing values (unseen categories) with global mean
        df[encoded_col_name] = df[encoded_col_name].fillna(global_mean)

        # Cleanup temporary key
        df.drop(columns=[temp_key], inplace=True)
        print(f"Created: {encoded_col_name}")

    return df, new_encoded_cols

import pandas as pd

def apply_catboost_encoding(df: pd.DataFrame, target: pd.Series, cols_to_encode: list) -> pd.DataFrame:
    """
    Encodes specified columns using CatBoost encoding and adds them to the dataframe 
    as new '_encoded' columns, preserving the originals and the index.
    """
    # Initialize the encoder
    encoder = CatBoostEncoder(cols=cols_to_encode)
    
    # Generate the encoded values
    # We fit_transform on the original columns using the target
    encoded_values = encoder.fit_transform(df[cols_to_encode], target)
    
    # Rename these new columns so they are clearly identified as the numeric versions
    encoded_values.columns = [f"{col}_encoded" for col in cols_to_encode]
    
    # Concatenate directly to the original dataframe
    # This preserves the original index and the original categorical columns
    return pd.concat([df, encoded_values], axis=1)



#=====
#
#EXAMPLE USAGE
#
#====


# --- Setup Sample Data ---
#data = pd.DataFrame({
#    'Y': [2.0, 5.0, 3.0, 8.0, 1.0, 6.0],
#    'Geo': ['US', 'CA', 'US', 'US', 'CA', 'CA'],
#    'Tier': ['Gold', 'Gold', 'Silver', 'Gold', 'Silver', 'Silver']
#})

# Scenario 1: Encode individual features
#df_encoded, encoded_cols = multi_ordered_target_encode(
#    df=data, 
#    cat_cols=['Geo', 'Tier'], 
#    target_col='Y', 
#    prior_weight=2.0
#)
#print("Encoded Columns:", encoded_cols)
# Outputs: ['Geo_catboost_encoded', 'Tier_catboost_encoded']

# Scenario 2: Encode individual features PLUS their unique intersection term
#df_combo, combo_cols = multi_ordered_target_encode(
#    df=data, 
#    cat_cols=['Geo', 'Tier'], 
#    target_col='Y', 
#    prior_weight=2.0,
#    encode_explicit_combinations=True
#)
#print("Combo Columns:", combo_cols)
# Outputs: ['Geo_catboost_encoded', 'Tier_catboost_encoded', 'Geo_X_Tier_catboost_encoded']


#===========
#EXPLICIT CORRECT VERSION 
#PIPELINE *NOT TESTED*
#==========


# 1. Generate a W matrix encoded against Y (for the outcome model)
#df_for_y, w_cols_y = multi_ordered_target_encode(
#    df=df, cat_cols=['W_high_card'], target_col='Y', prior_weight=5.0
#)

# 2. Generate a W matrix encoded against T (for the treatment model)
# Note: If T is binned/discrete, pass the collapsed 1D integer array T here
#df['T_integral'] = T
#df_for_t, w_cols_t = multi_ordered_target_encode(
#    df=df, cat_cols=['W_high_card'], target_col='T_integral', prior_weight=5.0
#)

# 3. Extract the encoded features
# (Include any other continuous confounders that don't need encoding)
#W_encoded_for_y = df_for_y[w_cols_y + ['W_continuous']].values
#W_encoded_for_t = df_for_t[w_cols_t + ['W_continuous']].values

# 4. Pass them directly to independent, pre-encoded pipelines
# This bypasses the need for EconML to handle the indices or the encoding step
#from sklearn.compose import ColumnTransformer

# Since the matrices are already fully numeric, the nuisance models 
# can just accept them directly as-is
#model_y_pipeline = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
#model_t_pipeline = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

# 5. Initialize the Estimator
#est = CausalForestDML(
#    model_y=model_y_pipeline,
#    model_t=model_t_pipeline,
#    discrete_treatment=True,
#    cv=5
#)

# 6. Fit using the split-encoded W syntax
# EconML allows you to pass specific features to model_y and model_t if needed,
# but passing the native CatBoost models directly (Option 2 from the previous response) 
# remains the cleanest way to automate this split logic internally.

