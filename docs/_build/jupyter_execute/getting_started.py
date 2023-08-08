#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd

content = load_iris()

data = pd.DataFrame(content.data, columns=[f"x{i+1}" for i in range(content.data.shape[1])])
data["y"] = (content.target == 1).astype(int)
print(data.head())


# In[2]:


from estyp.linear_model.stepwise import forward_selection, both_selection
from estyp.linear_model import LogisticRegression

formula = "y ~ x1 + x2 + x3 + x4"

ff1 = forward_selection(
    y       = "y",
    data    = data,
    model   = LogisticRegression,
    verbose = False,
)
ff2 = both_selection(
    formula = formula,
    data    = data,
    model   = LogisticRegression,
    verbose = False
)
print("- Forward result:", ff1)
print("- Both result   :", ff2)


# In[3]:


from estyp.testing import nested_models_test

model1 = LogisticRegression.from_formula(ff1, data).fit()
model2 = LogisticRegression.from_formula(ff2, data).fit()

nested_models_test(model1, model2) # First model is nested in the second one


# In[4]:


from estyp.testing import t_test

x = data["x1"]
y = data["x4"]

test_result = t_test(x, y)
print(test_result)


# In[5]:


from estyp.testing import var_test

test_result = var_test(x, y)
print(test_result)


# In[6]:


from estyp.testing import cor_test

test_result = cor_test(x, y, alternative="greater", method="spearman")
print(test_result)


# In[7]:


from estyp.testing import prop_test

counts = data["y"].value_counts()

test_result = prop_test(counts, p=0.75)
print(test_result)


# In[8]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[9]:


from estyp.cluster import NClusterSearch
from sklearn.cluster import KMeans

X = data.iloc[:, :-1].apply(lambda x: (x - x.mean()) / x.std())

searcher = NClusterSearch(
    estimator    = KMeans(n_init="auto"),
    method       = "elbow",
    random_state = 2023
)
searcher.fit(X)

print("- Clusters suggested: ", searcher.optimal_clusters_)
print("- Best estimator    : ", searcher.best_estimator_)
searcher.plot()


# In[10]:


from estyp.testing import CheckModel
import statsmodels.api as sm

model = sm.OLS.from_formula('x4 ~ x1 + x2 + x3', data=data).fit()
checker = CheckModel(model)
checker.check_all()

