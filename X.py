import pandas as pd
import numpy as np



def create_thresholded_heterogeneous_features(
    df: pd.DataFrame,
    categorical_cols: list,
    min_freq_threshold: float = 0.02,
    dtype: type = np.float64
):
    df_clean = df.copy()
    encoded_dfs = []
    
    # Trackers to separate model inputs from analysis columns
    ohe_feature_names = []
    mapped_text_names = []
    
    for col in categorical_cols:
        # 1. Frequency calculation
        freqs = df_clean[col].value_counts(normalize=True)
        rare_categories = freqs[freqs < min_freq_threshold].index.tolist()
        
        if isinstance(df_clean[col].dtype, pd.CategoricalDtype):
            if '_NONE_' not in df_clean[col].cat.categories:
                df_clean[col] = df_clean[col].cat.add_categories(['_NONE_'])
                
        # 2. Create the clean text column for later analysis
        mapped_col_name = f'{col}_mapped'
        df_clean[mapped_col_name] = df_clean[col].replace(rare_categories, '_NONE_')
        mapped_text_names.append(mapped_col_name)
        
        # 3. Create OHE binary features for the ML Model
        # This creates columns named: country_USA, country_Canada, country_Germany
        dummies = pd.get_dummies(
            df_clean[mapped_col_name], 
            prefix=col, 
            dtype=dtype
        )
        
        none_col = f"{col}__NONE_"
        if none_col in dummies.columns:
            dummies = dummies.drop(columns=[none_col])
        else:
            dummies = dummies.drop(columns=[dummies.columns[0]])
            
        encoded_dfs.append(dummies)
        ohe_feature_names.extend(dummies.columns.tolist())
        
    # Combine everything into the final dataframe
    result_df = pd.concat([df_clean] + encoded_dfs, axis=1)
    
    # Return the dataframe and the explicitly separated column lists
    return result_df, ohe_feature_names, mapped_text_names

'''
# Dummy data setup
data = pd.DataFrame({
    'country': ['USA', 'Canada', 'France', 'USA', 'Germany', 'Belize', 'Canada'],
    'treatment': [1, 0, 1, 1, 0, 0, 1],
    'confounder': [10, 20, 15, 12, 30, 45, 18],
    'outcome': [100, 110, 95, 105, 130, 85, 115]
})

# 1. Transform Data
df_processed, x_ohe_cols, analysis_text_cols = create_thresholded_heterogeneous_features(
    data, categorical_cols=['country'], min_freq_threshold=0.20
)

# --- Visual Inspection of what the function just returned ---
# x_ohe_cols         -> ['country_Canada', 'country_Germany', 'country_USA']
# analysis_text_cols -> ['country_mapped']

print("Columns sent to CausalForestDML (X):", x_ohe_cols)
print("Column used for CATE Groupby:", analysis_text_cols)

# 2. Extract Matrices for EconML
Y = df_processed['outcome'].values
T = df_processed['treatment'].values
W = df_processed[['confounder']].values

# HERE IS THE ANSWER TO YOUR QUESTION: 
# Pass ONLY the OHE column names array to the model as X
X = df_processed[x_ohe_cols].values 

# 3. Fit Model
cf = CausalForestDML(discrete_treatment=True, random_state=42)
cf.fit(Y, T, X=X, W=W)

# 4. Predict CATEs
# The model outputs a prediction for every single row
df_processed['CATE'] = cf.const_marginal_effect(X)

# 5. Compute Group CATEs using the text mapping column
# We grab the first element of our analysis list ('country_mapped')
group_column = analysis_text_cols[0] 
cate_summary = df_processed.groupby(group_column)['CATE'].mean()

print("\nFinal CATE Summary:")
print(cate_summary)


#############################
should allow effect() functionality to work without manual OHE, as long as you set
#############################

discrete_treatment=True,
categories='auto', # Or use [0, 1, 2] for explicit safety



#################################
# 1. Initialization
est = CausalForestDML(
    discrete_treatment=True,
    categories='auto', # Or use [0, 1, 2] for explicit safety
    model_t=RandomForestClassifier(n_estimators=100),
    model_y=..., # Your regressor
    n_estimators=1000
)

# 2. Fit (Pass raw 1D vector T, no need for manual OHE)
est.fit(Y, T, X=X)

# 3. Generating the "Relative CATE" Matrix
baseline = 0
treatment_levels = [1, 2, 3] # Adjust to your actual bins

cate_results = {}
for level in treatment_levels:
    # Gets the effect relative to your baseline
    cate_results[level] = est.effect(X_test, T0=baseline, T1=level)



#############################


'''


def create_thresholded_heterogeneous_features_old(
    df: pd.DataFrame,
    categorical_cols: list,
    min_freq_threshold: float = 0.02,
    dtype: type = np.float64
):
    df_clean = df.copy()
    encoded_dfs = []
    
    # Trackers to separate model inputs from analysis columns
    ohe_feature_names = []
    mapped_text_names = []
    
    for col in categorical_cols:
        # 1. Frequency calculation
        freqs = df_clean[col].value_counts(normalize=True)
        rare_categories = freqs[freqs < min_freq_threshold].index.tolist()
        
        if isinstance(df_clean[col].dtype, pd.CategoricalDtype):
            if '_NONE_' not in df_clean[col].cat.categories:
                df_clean[col] = df_clean[col].cat.add_categories(['_NONE_'])
                
        # 2. Create the clean text column for later analysis
        mapped_col_name = f'{col}_mapped'
        df_clean[mapped_col_name] = df_clean[col].replace(rare_categories, '_NONE_')
        mapped_text_names.append(mapped_col_name)
        
        # 3. Create OHE binary features for the ML Model
        # This creates columns named: country_USA, country_Canada, country_Germany
        dummies = pd.get_dummies(
            df_clean[mapped_col_name], 
            prefix=col, 
            dtype=dtype
        )
        
        none_col = f"{col}__NONE_"
        if none_col in dummies.columns:
            dummies = dummies.drop(columns=[none_col])
        else:
            dummies = dummies.drop(columns=[dummies.columns[0]])
            
        encoded_dfs.append(dummies)
        ohe_feature_names.extend(dummies.columns.tolist())
        
    # Combine everything into the final dataframe
    result_df = pd.concat([df_clean] + encoded_dfs, axis=1)
    
    # Return the dataframe and the explicitly separated column lists
    return result_df, ohe_feature_names, mapped_text_names

# =====================================================================
# STEP-BY-STEP WORKFLOW
# =====================================================================
'''
# Dummy data setup
data = pd.DataFrame({
    'country': ['USA', 'Canada', 'France', 'USA', 'Germany', 'Belize', 'Canada'],
    'treatment': [1, 0, 1, 1, 0, 0, 1],
    'confounder': [10, 20, 15, 12, 30, 45, 18],
    'outcome': [100, 110, 95, 105, 130, 85, 115]
})

# 1. Transform Data
df_processed, x_ohe_cols, analysis_text_cols = create_thresholded_heterogeneous_features(
    data, categorical_cols=['country'], min_freq_threshold=0.20
)

# --- Visual Inspection of what the function just returned ---
# x_ohe_cols         -> ['country_Canada', 'country_Germany', 'country_USA']
# analysis_text_cols -> ['country_mapped']

print("Columns sent to CausalForestDML (X):", x_ohe_cols)
print("Column used for CATE Groupby:", analysis_text_cols)

# 2. Extract Matrices for EconML
Y = df_processed['outcome'].values
T = df_processed['treatment'].values
W = df_processed[['confounder']].values

# HERE IS THE ANSWER TO YOUR QUESTION: 
# Pass ONLY the OHE column names array to the model as X
X = df_processed[x_ohe_cols].values 

# 3. Fit Model
cf = CausalForestDML(discrete_treatment=True, random_state=42)
cf.fit(Y, T, X=X, W=W)

# 4. Predict CATEs
# The model outputs a prediction for every single row
df_processed['CATE'] = cf.const_marginal_effect(X)

# 5. Compute Group CATEs using the text mapping column
# We grab the first element of our analysis list ('country_mapped')
group_column = analysis_text_cols[0] 
cate_summary = df_processed.groupby(group_column)['CATE'].mean()

print("\nFinal CATE Summary:")
print(cate_summary)
'''


def create_thresholded_heterogeneous_features_old(
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