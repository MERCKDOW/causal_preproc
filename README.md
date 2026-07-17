## **causal_preproc** ##
A collection of utilities designed to streamline preprocessing, modeling, and visualization workflows for causal analysis.
Key Modules
 * Encoding: Specialized functions for categorical variable processing.
 * Binning: Tools for discretizing continuous treatments.
 * Modeling:
   * Training Utilities: Helper functions for efficient model training routines.
   * Wrappers: Standardized interfaces for consistent model implementation.
 * Visualization: Tools for plotting diagnostics and analyzing results.
 * General Utilities: A library of helper functions for data manipulation and workflow management.





## **CausalForestDML: Methodology** ##

This module implements the Double Machine Learning (DML) framework to estimate heterogeneous treatment effects.
1. Nuisance Parameter Estimation (Pseudo-Residuals)
We first estimate the nuisance functions to partial out the effect of controls (W) and heterogeneity variables (X). We compute the pseudo-residuals as:
> Outcome Residuals:
> Y_tilde = Y - E[Y | X, W]
> Treatment Residuals:
> T_tilde = T - E[T | X, W]
> 
Where:
 * Y_tilde: Variation in Y not explained by X or W.
 * T_tilde: Variation in T not explained by X or W.
2. Interaction Model
With the residuals calculated, we isolate the treatment effect by regressing the pseudo-outcome on the pseudo-treatment, allowing the effect to vary across the heterogeneity features X:
> Model:
> Y_tilde = θ(X) * T_tilde + ε
> 
In Causal Forest, the model learns a non-parametric function θ(X), where the forest partitions the feature space X into leaves to estimate a distinct treatment effect within each segment.
3. Computation of CATE
The Conditional Average Treatment Effect (CATE), τ(X), represents the expected effect of the treatment on the outcome for an individual with characteristics X:
> CATE Definition:
> τ(X) = E[Y_tilde | X, T_tilde=1] - E[Y_tilde | X, T_tilde=0]
> Simplified Estimate:
> τ(X) = θ(X)
> 
When predicting for new samples, the Causal Forest averages the treatment effects from the relevant leaf nodes based on the input 




## **DynamicDML: Methodology** ##

DynamicDML estimates treatment effects in settings where treatments are applied sequentially or over time, accounting for time-varying confounders.
1. Sequential Nuisance Estimation
We estimate the expected outcome and treatment propensity at each time step (t) to generate residuals:
> Outcome Residuals:
> Y_tilde_t = Y_t - E[Y_t | X_t, W_t]
> Treatment Residuals:
> T_tilde_t = T_t - E[T_t | X_t, W_t]
> 
Where W_t represents time-varying controls and X_t represents state/heterogeneity variables.
2. Structural Interaction Model
We regress the residualized outcome on the residualized treatment to isolate the dynamic treatment effect:
> Model:
> Y_tilde_t = θ(X_t) * T_tilde_t + ε
> 
The function θ(X_t) represents the dynamic treatment effect conditional on the state X_t.
3. Computation of CATE
The dynamic Conditional Average Treatment Effect (CATE) is the marginal effect of the treatment on the outcome given the state:
> Dynamic CATE:
> τ(X_t) = ∂E[Y_t | X_t, T_t] / ∂T_t
> Estimation:
> τ(X_t) ≈ θ(X_t)
> 
Summary of Use
DynamicDML is utilized in environments such as dynamic pricing, inventory control, or personalized recommendation systems, where treatments are adaptive and confounders evolve alongside the state variables
