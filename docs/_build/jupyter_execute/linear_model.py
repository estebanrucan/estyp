#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from estyp.linear_model import LogisticRegression

np.random.seed(123)
data = pd.DataFrame({
   "y": np.random.randint(2, size=100),
   "x1": np.random.uniform(-1, 1, size=100),
   "x2": np.random.uniform(-1, 1, size=100),
})

formula = "y ~ x1 + x2"
spec = LogisticRegression.from_formula(formula, data)
model = spec.fit()

print(model.summary())

