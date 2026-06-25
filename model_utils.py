
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from econml.dml import CausalForestDML
from xgboost import XGBClassifier, XGBRegressor



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from econml.dml import CausalForestDML
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import OrdinalEncoder

def train_causal_models(
    df: pd.DataFrame,
    X_cols: List[str],
    W_cols: List[str],
    y_col: str,
    treatments_config: List[Dict[str, List[str]]],
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:

    trained_models = {}
    Y = df[y_col].values
    X = df[X_cols].values if X_cols else None
    W = df[W_cols].values if W_cols else None

    for treatment_dict in treatments_config:
        for treatment_name, cols in treatment_dict.items():
            # Logic fix: Handle single string column vs OHE columns
            if len(cols) == 1 and df[cols[0]].dtype == 'object':
                # It's a single categorical string column (like joint_treatment)
                print(f"Encoding categorical treatment: {cols[0]}")
                enc = OrdinalEncoder()
                T = enc.fit_transform(df[cols]).flatten().astype(int)
                ohe_cols_list = list(enc.categories_[0]) # Save category names for later mapping
            else:
                # It's a list of OHE columns
                T_ohe = df[cols].values
                T = np.argmax(T_ohe, axis=1)
                ohe_cols_list = cols

            model = CausalForestDML(
                model_y=XGBRegressor(random_state=random_state),
                model_t=XGBClassifier(random_state=random_state),
                discrete_treatment=True,
                random_state=random_state
            )

            model.fit(Y, T, X=X, W=W)

            trained_models[treatment_name] = {
                'model': model,
                'ohe_cols': ohe_cols_list
            }

    return trained_models
# Re-run training with the joint treatment column name passed as a list
from sklearn.calibration import CalibratedClassifierCV
# Calibrate the propensity model to prevent overlap failure
#model_t_calibrated = CalibratedClassifierCV(
#    estimator=XGBClassifier(max_depth=4, min_child_weight=20, subsample=0.8),
#    method='isotonic',
#    cv=5
#)

# Robust outcome model
#model_y_robust = XGBRegressor(max_depth=5, min_child_weight=15, subsample=0.8)

def train_causal_models_joint_treatment(
    df: pd.DataFrame,
    X_cols: List[str],
    W_cols: List[str],
    y_col: str,
    treatments_config: List[Dict[str, List[str]]],
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:

    trained_models = {}
    Y = df[y_col].values
    X = df[X_cols].values if X_cols else None
    W = df[W_cols].values if W_cols else None

    for treatment_dict in treatments_config:
        for treatment_name, cols in treatment_dict.items():
            # Logic fix: Handle single string column vs OHE columns
            if len(cols) == 1 and df[cols[0]].dtype == 'object':
                # It's a single categorical string column (like joint_treatment)
                print(f"Encoding categorical treatment: {cols[0]}")
                enc = OrdinalEncoder()
                T = enc.fit_transform(df[cols]).flatten().astype(int)
                ohe_cols_list = list(enc.categories_[0]) # Save category names for later mapping
            else:
                # It's a list of OHE columns
                T_ohe = df[cols].values
                T = np.argmax(T_ohe, axis=1)
                ohe_cols_list = cols

            model = CausalForestDML(
                #model_y=XGBRegressor(random_state=random_state),
                #model_t=XGBClassifier(random_state=random_state),
                model_y=XGBRegressor(max_depth=5, min_child_weight=15, subsample=0.8),
                model_t=XGBClassifier(max_depth=4, min_child_weight=20, subsample=0.8),
                #model_t=CalibratedClassifierCV(estimator=XGBClassifier(max_depth=4, min_child_weight=20, subsample=0.8),
                #                                method='isotonic',
                #                                cv=5
                #                              ),

                discrete_treatment=True,
                n_estimators=2000,
                min_samples_leaf=50,
                cv=10,
                random_state=random_state
            )

            model.fit(Y, T, X=X, W=W)

            trained_models[treatment_name] = {
                'model': model,
                'ohe_cols': ohe_cols_list
            }

    return trained_models



def train_causal_models_does_not_work(
    df: pd.DataFrame,
    X_cols: List[str],
    W_cols: List[str],
    y_col: str,
    treatments_config: List[Dict[str, List[str]]],
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Trains a CausalForestDML model for each treatment defined in the configuration.

    Args:
        df: The main pandas DataFrame.
        X_cols: List of column names for heterogeneous features.
        W_cols: List of column names for control features.
        y_col: The target variable column name.
        treatments_config: List of dictionaries mapping treatment names to their OHE columns.
                           Example: [{'discount': ['disc_0', 'disc_10', 'disc_20']}, ...]
        random_state: Integer for reproducibility.

    Returns:
        Dictionary mapping treatment names to their trained model and OHE column list.
    """
    trained_models = {}
    
    Y = df[y_col].values
    X = df[X_cols].values if X_cols else None
    W = df[W_cols].values if W_cols else None

    for treatment_dict in treatments_config:
        for treatment_name, ohe_cols in treatment_dict.items():
            
            # Convert OHE to a 1D categorical integer array for EconML
            T_ohe = df[ohe_cols].values
            T = np.argmax(T_ohe, axis=1)
            print(type(T))
            # Initialize the DML model
            # XGBClassifier handles the discrete treatment prediction (propensity)
            # XGBRegressor handles the target outcome prediction
            model = CausalForestDML(
                model_y=XGBRegressor(random_state=random_state),
                model_t=XGBClassifier(random_state=random_state),
                discrete_treatment=True,
                n_estimators=2000,
                min_samples_leaf=50,
                cv=10,                
                random_state=random_state
            )

            model.fit(Y, T, X=X, W=W)

            trained_models[treatment_name] = {
                'model': model,
                'ohe_cols': ohe_cols
            }
            
    return trained_models



def compute_and_aggregate_cates(
    df: pd.DataFrame, 
    X_cols: List[str], 
    heterogeneous_cat_cols: List[str], 
    trained_models: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Computes CATEs for all treatment levels and aggregates them by specified heterogeneous variables.

    Args:
        df: The main pandas DataFrame.
        X_cols: List of column names for heterogeneous features used during training.
        heterogeneous_cat_cols: Categorical columns in the dataframe to group by.
        trained_models: Output from `train_causal_models`.

    Returns:
        Dictionary mapping each heterogeneous column to its aggregated DataFrame 
        and the ordered list of treatment columns.
    """
    X = df[X_cols].values
    
    # DataFrame to temporarily hold row-level CATE predictions
    cates_df = df[heterogeneous_cat_cols].copy()
    treatment_order = [] 

    for treatment_name, model_info in trained_models.items():
        model = model_info['model']
        ohe_cols = model_info['ohe_cols']

        # Skip index 0 as it is the control group (T0=0)
        for i in range(1, len(ohe_cols)):
            treatment_level_name = ohe_cols[i]
            treatment_order.append((treatment_name, treatment_level_name))
            
            # Compute conditional average treatment effect
            cates_df[treatment_level_name] = model.effect(X, T0=0, T1=i)

    # Aggregate by the specified heterogeneous categorical variables
    aggregated_results = {}
    ordered_level_names = [t[1] for t in treatment_order]

    for cat_col in heterogeneous_cat_cols:
        # Calculate the mean CATE for each level of the categorical feature
        agg_df = cates_df.groupby(cat_col)[ordered_level_names].mean()
        
        aggregated_results[cat_col] = {
            'data': agg_df,
            'treatment_order': treatment_order 
        }

    return aggregated_results