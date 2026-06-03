import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

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
            tree = DecisionTreeRegressor(max_leaf_nodes=k, random_state=42)
            scores = cross_val_score(tree, X_trans, y_trans, cv=5, scoring='r2')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        
        optimal_tree = DecisionTreeRegressor(max_leaf_nodes=best_k, random_state=42)
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