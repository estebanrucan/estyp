from statsmodels.api import GLM, OLS
from pandas import DataFrame
from estyp.linear_model.stepwise.__base import __both_selection, __forward_selection

def both_selection(formula: str, data: DataFrame, model: GLM, max_iter = 10000, **kwargs) -> str:
    """
# Both Forward and Backward Variable Selection for GLM's

This function performs both forward and backward variable selection using the Akaike Information Criterion (AIC).

## Parameters

* `formula`: A string representing the initial model formula.
* `data`: A Pandas DataFrame containing the data to be used for model fitting.
* `model`: A statsmodels.GLM object that represents the type of model to be fit.
* `max_iter`: The maximum number of iterations to perform.
* `**kwargs`: Additional keyword arguments to be passed to the model.from_formula() method.

## Returns

A string representing the final model formula.

## Example

```python
import statsmodels.api as sm
import pandas as pd

data = pd.DataFrame({
    "y": [1, 2, 3, 4, 5],
    "x1": [1, 2, 3, 4, 5],
    "x2": [6, 7, 8, 9, 10],
})

formula = "y ~ x1 + x2"

model = sm.OLS

final_formula = both_selection(formula=formula, data=data, model=model)

print(final_formula)
    """
    return __both_selection(formula, data, model, max_iter, **kwargs)

def forward_selection(
    y: str, 
    data: DataFrame,
    model: GLM = OLS, 
    alpha = 0.05,
    **kwargs
):
    """
# Forward Variable Selection for GLM's

This function performs forward variable selection using p-values calculated from nested models testing.

## Parameters

* `y`: A string containing the name of the dependent variable (target) to be predicted.
* data: The pandas DataFrame containing both the target variable 'y' and the        predictor variables for model training.
* `model`: A statsmodels model class. The statistical model to be used for model fitting and evaluation. Defaults to `sm.OLS`.
* `alpha`: A number between 0 and 1. The significance level for feature selection. A feature is added to the model if its p-value is less than this alpha value. Defaults to 0.05.
* `**kwargs`: Additional keyword arguments to be passed to the model.from_formula() method.

## Returns

A string representing the final model formula.

## Example

```python
import pandas as pd
import statsmodels.api as sm

# Create sample DataFrame
data = pd.DataFrame({
    'y': [1, 2, 3, 4, 5],
    'X1': [2, 4, 5, 7, 9],
    'X2': [3, 1, 6, 8, 4],
    'X3': [1, 5, 9, 2, 3]
})

# Perform the forward variable selection
formula = forward_selection(
    y = "y", ,
    data = data, 
    model = sm.OLS, 
    alpha = 0.05
)

# Fit the model using the selected formula
selected_model = sm.OLS.from_formula(formula, data).fit()
print(selected_model.summary())
```
    """
    return __forward_selection(y, data, model, alpha, **kwargs)