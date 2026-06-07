import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import silhouette_score
import warnings

def create_optimal_binned_treatment(
    df: pd.DataFrame, 
    treatment_col: str, 
    bin_candidates: list = [3, 5, 6], 
    target_col: str = None, 
    drop_first: bool = True,
    dtype: type = np.float64
):
    df_clean = df.copy()
    X_trans = df_clean[[treatment_col]].to_numpy().astype(dtype)
    best_k = bin_candidates[0]
    
    if target_col is not None:
        y_trans = df_clean[target_col].to_numpy().astype(dtype)
        best_score = -float('inf')
        
        for k in bin_candidates:
            tree = DecisionTreeRegressor(max_leaf_nodes=k,
                                         min_samples_leaf=0.05,
                                         random_state=42)
            scores = cross_val_score(tree, X_trans, y_trans, cv=5, scoring='r2')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        
        optimal_tree = DecisionTreeRegressor(max_leaf_nodes=best_k,
                                             min_samples_leaf=0.05,
                                             random_state=42)
        optimal_tree.fit(X_trans, y_trans)
        bin_assignments = optimal_tree.apply(X_trans)
        unique_leaves = np.unique(bin_assignments)
        leaf_means = [X_trans[bin_assignments == leaf].mean() for leaf in unique_leaves]
        ordered_leaves = [leaf for _, leaf in sorted(zip(leaf_means, unique_leaves))]
        leaf_to_bin_map = {leaf: f"{treatment_col}_bin_{i}" for i, leaf in enumerate(ordered_leaves)}
        bin_labels = [leaf_to_bin_map[leaf] for leaf in bin_assignments]
        
        df_bins = pd.DataFrame({f"{treatment_col}_bin": bin_labels}, index=df_clean.index)
        one_hot = pd.get_dummies(df_bins[f"{treatment_col}_bin"], dtype=dtype)
        one_hot = one_hot[[f"{treatment_col}_bin_{i}" for i in range(best_k)]]
    else:
        best_k = max(bin_candidates)
        quantile_labels = [f"{treatment_col}_bin_{i}" for i in range(best_k)]
        try:
            binned_series = pd.qcut(df_clean[treatment_col], q=best_k, labels=quantile_labels, duplicates='drop')
            one_hot = pd.get_dummies(binned_series, dtype=dtype)
        except ValueError:
            binned_series = pd.cut(df_clean[treatment_col], bins=best_k, labels=quantile_labels)
            one_hot = pd.get_dummies(binned_series, dtype=dtype)

    new_column_names = list(one_hot.columns)
    if drop_first:
        reference_col = new_column_names[0]
        one_hot = one_hot.drop(columns=[reference_col])
        new_column_names.remove(reference_col)
        
    return pd.concat([df, one_hot], axis=1), new_column_names



def optimal_binning_dml(df, columns, target=None, bin_candidates=(3, 4, 5), frac=0.2, drop_zero_bin=True, random_state=42):
    """
    Optimally bins continuous treatments, guarantees a dedicated 0.0 baseline bin,
    and returns one-hot encoded variables. Optionally drops the 0.0 bin for DynamicDML.

    Parameters:
    -----------
    df : pandas.DataFrame
        The full dataset.
    columns : str or list
        The treatment column(s) to bin.
    target : str, optional
        The target column (e.g., 'revenue') for supervised binning. Default is None.
    bin_candidates : tuple
        Candidate numbers of bins to test for the positive (>0) data.
    frac : float
        Fraction of data to use for finding optimal edges to prevent leakage.
    drop_zero_bin : bool
        If True, drops the baseline '0.0' OHE column to prevent the dummy variable trap in DynamicDML.
    random_state : int
        Seed for reproducibility.

    Returns:
    --------
    df_updated : pandas.DataFrame
        The original dataframe with the new one-hot encoded columns appended (and baseline dropped if requested).
    new_ohe_cols : list
        A list of the newly created and kept one-hot encoded column names.
    """
    df_updated = df.copy()
    if isinstance(columns, str):
        columns = [columns]

    new_ohe_cols = []
    epsilon = 1e-9 # To safely isolate 0.0 from strictly positive treatments

    for col in columns:
        # 1. Sample the data to prevent leakage
        sample_idx = df_updated.sample(frac=frac, random_state=random_state).index
        X_sample_full = df_updated.loc[sample_idx, [col]].copy().dropna()

        # Isolate strictly positive data for optimal binning optimization
        X_sample_pos = X_sample_full[X_sample_full[col] > epsilon]

        best_k = None
        best_score = -np.inf
        best_pos_edges = []

        # Check if we have enough positive data to optimize splits
        if len(X_sample_pos) > len(bin_candidates) * 2:
            if target:
                # SUPERVISED: Optimize splits on positive data relative to target
                y_sample_pos = df_updated.loc[X_sample_pos.index, target]

                for k in bin_candidates:
                    dt = DecisionTreeRegressor(max_leaf_nodes=k,
                                               min_samples_leaf=0.05,
                                               random_state=random_state)
                    
                    dt.fit(X_sample_pos, y_sample_pos)

                    thresholds = dt.tree_.threshold
                    edges = np.unique(thresholds[thresholds != -2.0])
                    edges = sorted(list(edges))

                    if len(edges) == 0:
                        continue

                    # Evaluate this split choice
                    eval_edges = [epsilon] + edges + [np.inf]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        binned_eval = pd.cut(X_sample_pos[col], bins=eval_edges, labels=False, duplicates='drop')

                    if binned_eval.nunique() <= 1:
                        continue

                    score = mutual_info_regression(binned_eval.values.reshape(-1, 1), y_sample_pos, random_state=random_state)[0]

                    if score > best_score:
                        best_score = score
                        best_pos_edges = edges
            else:
                # UNSUPERVISED: Maximize Silhouette Score using KMeans on positive data
                for k in bin_candidates:
                    kbd = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans', random_state=random_state)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        binned_eval = kbd.fit_transform(X_sample_pos).flatten()

                    if len(np.unique(binned_eval)) > 1:
                        score = silhouette_score(X_sample_pos, binned_eval, random_state=random_state)
                    else:
                        score = -1

                    if score > best_score:
                        best_score = score
                        # Drop the first and last edges generated by KMeans; we manage boundaries manually
                        best_pos_edges = list(kbd.bin_edges_[0])[1:-1]

        # Construct final edges: Explicitly isolate (-inf, 0.0], then insert optimized positive splits
        final_edges = [-np.inf, epsilon] + sorted(best_pos_edges) + [np.inf]

        # 2. Apply final engineered edges to the ENTIRE dataset
        binned_series = pd.cut(df_updated[col], bins=final_edges, duplicates='drop')

        # 3. Create One-Hot Encoded variables
        dummies = pd.get_dummies(binned_series, prefix=col, dummy_na=False).astype(int)

        # 4. Clean column names to make them safe for ML models
        clean_cols = {}
        baseline_col_name = None

        for c in dummies.columns:
            clean_name = str(c).replace(', ', '_to_').replace('(', '').replace('[', '').replace(']', '').replace('.0_to_', '_to_')
            # Identify the specific baseline column containing the zeros
            if '-inf_to_1e-09' in clean_name or '-inf_to_0.0' in clean_name:
                clean_name = f"{col}_baseline_0.0"
                baseline_col_name = clean_name
            else:
                # Clean up ugly scientific notation/epsilon string artifacts for positive bins
                clean_name = clean_name.replace('1e-09', '0.0')
            clean_cols[c] = clean_name

        dummies = dummies.rename(columns=clean_cols)

        # 5. Drop the zero bin if tailored for DynamicDML / Linear DML models
        if drop_zero_bin and baseline_col_name in dummies.columns:
            dummies = dummies.drop(columns=[baseline_col_name])

        # Concat to our main dataframe and log feature names
        new_ohe_cols.extend(dummies.columns.tolist())
        df_updated = pd.concat([df_updated, dummies], axis=1)

    return df_updated, new_ohe_cols


#-------
#DYNAMIC USAGE
#-------

# drop_zero_bin=True is the default
#df_dynamic, dynamic_treatments = optimal_binning_dml(
#    df=df,
#    columns=['discount', 'shipping_time', 'frustration_score'],
#    target='revenue',
#    bin_candidates=(3, 4, 5),
#    drop_zero_bin=True
#)

# 'dynamic_treatments' will look like:
# ['discount_0.0_to_0.25', 'discount_0.25_to_inf', 'shipping_time_0.0_to_3.0', ...]
# Notice there is no 'discount_baseline_0.0' column here.



#-------
#CAUSAL USAGE
#-------


#    df_forest, forest_treatments = optimal_binning_dml(
#    df=df,
#    columns=['discount', 'shipping_time', 'frustration_score'],
#    target='revenue',
#    bin_candidates=(3, 4, 5),
#    drop_zero_bin=False # Keeps the 0.0 column intact
#)

# 'forest_treatments' will contain:
# ['discount_baseline_0.0', 'discount_0.0_to_0.25', 'discount_0.25_to_inf', ...]

