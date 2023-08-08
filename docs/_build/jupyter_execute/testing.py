#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[2]:


import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from estyp.testing import CheckModel

diabetes = load_diabetes()
X = diabetes["data"]
y = diabetes["target"]
X = sm.add_constant(X)
model = sm.OLS(y, X)
fitted_model = model.fit()
cm = CheckModel(fitted_model)
cm.check_all()


# In[3]:


import numpy as np
from estyp.testing import var_test

np.random.seed(2023)
x = np.random.normal(size=100)
y = np.random.normal(size=100)

print("1 - F Test for Two Samples")
print(var_test(x, y))
print("2 - F Test for Two Samples changing alternative hypothesis")
print(var_test(x, y, alternative="less"))
print("3 - F Test for Two Samples changing ratio")
print(var_test(x, y, ratio=0.9, alternative="greater"))


# In[4]:


import numpy as np
from estyp.testing import t_test

np.random.seed(2023)
x = np.random.normal(size=100)
y = np.random.normal(size=100)
mu = 0.1

print("1 - One Sample Test")
print(t_test(x, mu=mu, alternative="less"))
print("2 - Two Sample Test")
print(t_test(x, y, mu=mu))
print("3 - Two Sample Test with Equal Variances")
print(t_test(x, y, mu=mu, var_equal=True, alternative="greater"))
print("4 - Paired Test")
print(t_test(x, y, mu=mu, paired=True))


# In[5]:


import pandas as pd
import statsmodels.api as sm
from estyp.testing import nested_models_test

data = pd.DataFrame({
    "x": [2.01, 2.99, 4.01, 5.01, 6.89],
    "y": [2, 3, 4, 5, 6]
})
model_small = sm.OLS.from_formula("y ~ 1", data).fit()
model_big = sm.OLS.from_formula("y ~ x", data).fit()
print(nested_models_test(model_small, model_big))


# In[6]:


data = pd.DataFrame({
    "x": [2.01, 2.99, 4.01, 3.01, 4.89],
    "y": [0, 1, 1, 0, 1]
})
model_small = sm.Logit.from_formula("y ~ 1", data).fit()
model_big = sm.Logit.from_formula("y ~ x", data).fit()
print(nested_models_test(model_small, model_big))


# In[7]:


data = pd.DataFrame({
    "x": [2.01, 2.99, 4.01, 5.01, 6.89],
    "y": [2, 3, 4, 5, 6]
})
model_small = sm.GLM.from_formula("y ~ 1", data, family = sm.families.Gamma()).fit()
model_big = sm.GLM.from_formula("y ~ x", data, family = sm.families.Gamma()).fit()
print(nested_models_test(model_small, model_big))


# In[8]:


import numpy as np
from scipy import stats
from estyp.testing import prop_test

x = np.array([83, 90, 129, 70])
n = np.array([86, 93, 136, 82])
result = prop_test(x, n)
print(result)


# In[9]:


from sklearn import datasets
from estyp.testing import cor_test

iris = datasets.load_iris()
sepal_length = iris.data[:, 0]
petal_length = iris.data[:, 2]

result = cor_test(sepal_length, petal_length, method="pearson")
print(result)


# In[10]:


from estyp.testing import chisq_test

M = [[762, 327, 468], [484, 239, 477]]
result1 = chisq_test(M)
print(result1)


# In[11]:


x = [12, 5, 7, 7]
p = [0.4, 0.4, 0.2, 0.2]
result2 = chisq_test(x, p=p, rescale_p=True)
print(result2)


# In[12]:


x = [20, 15, 25]
result31 = chisq_test(x)
print(result31)


# In[13]:


x = [89,37,30,28,2]
p = [0.40,0.20,0.20,0.19,0.01]
result32 = chisq_test(x, p=p)
print(result32)


# In[14]:


x = [1, 2, 3, 4, 5, 6]
y = [6, 1, 2, 3, 4, 5]
result4 = chisq_test(x, y)
print(result4)

