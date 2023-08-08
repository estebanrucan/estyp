#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[2]:


from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from estyp.cluster import NClusterSearch

data = load_iris().data
new_data = load_iris().data[:10]
searcher = NClusterSearch(estimator=KMeans(), method='elbow')
searcher.fit(data)

labels = searcher.labels_
predicted_labels = searcher.predict(new_data)
optimal_model = searcher.best_estimator_
optimal_clusters = searcher.optimal_clusters_

searcher.plot()


# In[3]:


from kmodes.kmodes import KModes
import pandas as pd
import numpy as np

np.random.seed(2023)
data = pd.DataFrame(np.random.randint(0, 10, size=(100, 4))).apply(lambda x: x.astype('object'))

kmodes = KModes(init='Huang', n_init=5)
searcher = NClusterSearch(estimator=kmodes, method='elbow')
searcher.fit(data)
searcher.plot()


# In[4]:


from sklearn.datasets import load_iris
from sklearn_extra.cluster import KMedoids

data = load_iris().data

searcher = NClusterSearch(estimator=KMedoids(), method='silhouette')
searcher.fit(data)
searcher.plot()


# In[5]:


import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes

np.random.seed(2023)
data = pd.DataFrame(np.random.randint(0, 10, size=(100, 4))).apply(lambda x: x.astype('object'))
data["new"] = np.random.randint(0, 10, size=(100, 1))

searcher = NClusterSearch(estimator=KPrototypes(), method='silhouette', verbose=True)
searcher.fit(data, categorical=[0, 1, 2, 3])
searcher.plot()


# In[6]:


import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes

np.random.seed(2023)
data = pd.DataFrame(np.random.randint(0, 10, size=(100, 4))).apply(lambda x: x.astype('object'))
data["new"] = np.random.randint(0, 10, size=(100, 1))

searcher = NClusterSearch(estimator=KPrototypes(), method='silhouette')
searcher.fit(data, categorical=[0, 1, 2, 3], alpha=0.1)
searcher.plot()

