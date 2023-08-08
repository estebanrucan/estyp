#!/usr/bin/env python
# coding: utf-8

# In[1]:


import statsmodels.api as sm
import pandas as pd
from estyp.linear_model.stepwise import both_selection

data = pd.DataFrame({
    "y": [1, 2, 3, 4, 5],
    "x1": [1, 2, 3, 4, 5],
    "x2": [6, 7, 8, 9, 10],
})
formula = "y ~ x1 + x2"
model = sm.OLS

final_formula = both_selection(formula=formula, data=data, model=model)
print(final_formula)


# In[2]:


import pandas as pd
import statsmodels.api as sm
from estyp.linear_model.stepwise import forward_selection

# Create sample DataFrame
data = pd.DataFrame({
    'y': [1, 2, 3, 4, 5],
    'X1': [2, 4, 5, 7, 9],
    'X2': [3, 1, 6, 8, 4],
    'X3': [1, 5, 9, 2, 3]
})

# Perform the forward variable selection
formula = forward_selection(
    y = "y",
    data = data,
    model = sm.OLS,
    alpha = 0.05
)

# Fit the model using the selected formula
selected_model = sm.OLS.from_formula(formula, data).fit()
print(selected_model.summary())

