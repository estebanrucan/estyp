import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from kmodes.kprototypes import matching_dissim, euclidean_dissim
from sklearn.metrics import silhouette_score
from typing import Literal
from joblib import Parallel, delayed


class NClusterSearch:
    """
Optimal Number of Clusters Searcher
====================================

View this in the [online documentation](https://estyp.readthedocs.io/en/latest/cluster.html#optimal-number-of-clusters-searcher).

Description
-----------

A helper class to identify the optimal number of clusters for clustering algorithms.

This class offers methods like the elbow method and silhouette method to evaluate
and suggest the best number of clusters for the provided data. The class currently
supports KMeans, KMedoids, KModes, and KPrototypes as estimators.

Parameters:
-----------
`estimator` : instance of clustering model
    The clustering algorithm for which you want to find the optimal number of clusters. Also can be a Pipeline object with the last step being a clustering algorithm.
    Supported algorithms: `KMeans`, `KMedoids`, `KModes`, `KPrototypes`.

`method` : str, optional (default = 'elbow')
    The method used to determine the optimal number of clusters.
    Accepted values: 'elbow', 'silhouette'.

`min_clusters` : int, optional (default = 1)
    The minimum number of clusters to consider.

`max_clusters` : int, optional (default = 10)
    The maximum number of clusters to consider.

`step` : int, optional (default = 1)
    The step size for increasing the number of clusters during the search.

`random_state` : int, optional (default = 123)
    Random seed for reproducibility.
    
`n_jobs` : int, optional (default = 1)
    The number of jobs to run in parallel for the search. If -1, then the number of jobs is set to the number of CPU cores.

`verbose` : bool, optional (default = False)
    If True, the process will print details as it proceeds.

Examples:
---------

Example 1: Using the elbow method with KMeans

>>> from sklearn.cluster import KMeans
>>> from sklean.datasets import load_iris
>>> data = load_iris().data
>>> new_data = load_iris().data[:10]
>>> searcher = NClusterSearch(estimator=KMeans(), method='elbow')
>>> searcher.fit(data)
>>> searcher.plot()
>>> labels = searcher.labels_
>>> predicted_labels = searcher.predict(new_data)
>>> optimal_model = searcher.best_estimator_
>>> optimal_clusters = searcher.optimal_clusters_


Example 2: Using KModes with custom arguments

>>> from kmodes.kmodes import KModes
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(2023)
>>> data = pd.DataFrame(np.random.randint(0, 10, size=(100, 4))).apply(lambda x: x.astype('object'))
>>> kmodes = KModes(init='Huang', n_init=5, verbose=1)
>>> searcher = NClusterSearch(estimator=kmodes, method='elbow')
>>> searcher.fit(data)

Example 3: Using the silhouette method with KMedoids

>>> from sklean.datasets import load_iris
>>> from sklearn_extra.cluster import KMedoids
>>> data = load_iris().data
>>> searcher = NClusterSearch(estimator=KMedoids(), method='silhouette')
>>> searcher.fit(data)
>>> searcher.plot()

Example 4: Using the silhouette method with KPrototypes

>>> import pandas as pd
>>> import numpy as np
>>> from kmodes.kprototypes import KPrototypes
>>> np.random.seed(2023)
>>> data = pd.DataFrame(np.random.randint(0, 10, size=(100, 4))).apply(lambda x: x.astype('object'))
>>> data["new"] = np.random.randint(0, 10, size=(100, 1))
>>> searcher = NClusterSearch(estimator=KPrototypes(), method='silhouette', verbose=True)
>>> searcher.fit(data, categorical=[0, 1, 2, 3])
>>> searcher.plot()

Example 5: Using the silhouette method with KPrototypes and custom alpha for distance calculation

>>> import pandas as pd
>>> import numpy as np
>>> from kmodes.kprototypes import KPrototypes
>>> np.random.seed(2023)
>>> data = pd.DataFrame(np.random.randint(0, 10, size=(100, 4))).apply(lambda x: x.astype('object'))
>>> data["new"] = np.random.randint(0, 10, size=(100, 1))
>>> searcher = NClusterSearch(estimator=KPrototypes(), method='silhouette')
>>> searcher.fit(data, categorical=[0, 1, 2, 3], alpha=0.1)
>>> searcher.plot()

Example 6: Using the elbow method with a pipeline of with standard scaler and Kprototypes and parallel processing.

>>> import pandas as pd
>>> from sklearn.datasets import load_iris
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.compose import make_column_transformer
>>> from sklearn.pipeline import make_pipeline
>>> from kmodes.kprototypes import KPrototypes
>>> from estyp.cluster import NClusterSearch
>>> X, y = load_iris(return_X_y=True)
>>> df = pd.DataFrame(X)
>>> df.columns = df.columns.astype(str)
>>> df["y"] = y
>>> transformer = make_column_transformer(
...     (StandardScaler(), [0, 1, 2, 3]),
...     remainder='passthrough'
... )
>>> kproto = KPrototypes(init='Huang')
>>> model = make_pipeline(transformer, kproto)
>>> searcher = NClusterSearch(
...     estimator    = model,
...     method       = "elbow",
...     min_clusters = 1,
...     max_clusters = 10,
...     n_jobs       = -1
... )
>>> searcher.fit(df, kprototypes__categorical = [4])
>>> searcher.plot() 
    """

    def __init__(
        self,
        estimator,
        method: Literal["elbow", "silhouette"] = "elbow",
        *,
        min_clusters: int = 1,
        max_clusters: int = 10,
        step: int = 1,
        random_state: int = 123,
        n_jobs=1,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.method = method
        self.min_clusters = min_clusters
        if self.method == "silhouette" and (self.min_clusters == 1):
            self.min_clusters = 2
        self.max_clusters = max_clusters
        self.step = step
        self.random_state = random_state
        self.__range = list(
            range(self.min_clusters, self.max_clusters + 1, self.step))
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.__is_pipeline = type(self.estimator).__name__ == "Pipeline"

        self.__model_name = type(
            self.estimator[-1]).__name__ if self.__is_pipeline else type(self.estimator).__name__
        if self.__model_name in ["KMeans", "KMedoids"]:
            self.__cost = "inertia_"
        elif self.__model_name in ["KModes", "KPrototypes"]:
            self.__cost = "cost_"
        else:
            raise ValueError(
                f"The estimator \"{self.__model_name}\" is not implemented yet."
            )

        if self.min_clusters > self.max_clusters:
            raise Exception(
                f"The minimum number of clusters ({self.min_clusters}) is greater than the maximum number of clusters ({self.max_clusters})."
            )

        if self.estimator.__dict__.get("random_state") is None:
            if self.__is_pipeline:
                self.estimator[-1].set_params(random_state=self.random_state)
            else:
                self.estimator.set_params(random_state=self.random_state)

        if self.method not in ["elbow", "silhouette"]:
            raise Exception(
                f"The method '{self.method}' is not implemented yet.")

    def fit(self, X, **kwargs):
        """
Fits the estimator with the data over a range of cluster numbers.

Parameters:
-----------
`X` : array-like, shape (n_samples, n_features)
    The data to determine the optimal number of clusters for.

`**kwargs` : Additional keyword arguments to be passed to the estimator's fit method.

Returns:
--------
`self` : object
    Returns an instance of `NClusterSearch()`.

Example:
--------

>>> from sklearn.cluster import KMeans
>>> from sklean.datasets import load_iris
>>> data = load_iris().data
>>> searcher = NClusterSearch(estimator=KMeans(), method='elbow')
>>> searcher.fit(data)
        """
        self.__X = X

        if self.__X.shape[0] < self.max_clusters:
            raise Exception(
                f"The number of samples ({self.__X.shape[0]}) is lower than the maximum number of clusters ({self.max_clusters})."
            )
        if self.__X.shape[0] < self.min_clusters:
            raise Exception(
                f"The number of samples ({self.__X.shape[0]}) is lower than the minimum number of clusters ({self.max_clusters})."
            )

        self.__kwargs = kwargs

        if self.method == "elbow":
            self.__elbow_method()
        elif self.method == "silhouette":
            self.__silhuette_method()
        self.__fitted = True

        return self

    @property
    def best_estimator_(self):
        """
Returns the estimator with the optimal number of clusters.

This property can be accessed only after calling the fit method.

Returns:
--------
best_estimator : instance of clustering model
    Estimator set to the optimal number of clusters and trained with the data.

Example:
--------

>>> from sklearn.cluster import KMeans
>>> from sklean.datasets import load_iris
>>> data = load_iris().data
>>> model = NClusterSearch(estimator=KMeans(), method='elbow')
>>> model.fit(data)
>>> optimal_model = model.best_estimator_
        """
        if not self.__fitted:
            raise Exception(
                "The model is not fitted yet. Call 'fit' with appropriate arguments before using this property."
            )
        if self.__is_pipeline:
            self.estimator[-1].set_params(n_clusters=self.optimal_clusters_)
            return self.estimator.fit(self.__X, **self.__kwargs)
        else:
            return self.estimator.set_params(n_clusters=self.optimal_clusters_).fit(
                self.__X, **self.__kwargs
            )

    def __compute_elbow_point(self, sse):
        # Calculate the line from first to last point
        x = np.arange(len(sse))
        y = np.array(sse)
        line_k = (y[-1] - y[0]) / (x[-1] - x[0])
        line_b = y[0] - line_k * x[0]

        # Calculate distance from each point to the line
        line_y = line_k * x + line_b
        distance = np.abs(line_y - y) / np.sqrt(1 + line_k**2)

        # Return the index of point with max distance
        return np.argmax(distance)

    def _elbow_method_for_content(self, n_clusters):
        if self.__is_pipeline:
            self.estimator[-1].set_params(n_clusters=n_clusters)
            self.estimator.fit(self.__X, **self.__kwargs)
            return self.estimator[-1].__getattribute__(self.__cost)
        else:
            self.estimator.set_params(n_clusters=n_clusters)
            self.estimator.fit(self.__X, **self.__kwargs)
            return self.estimator.__getattribute__(self.__cost)

    def __elbow_method(self):
        if self.verbose:
            print("Calculating costs...")
        self.__costs = []
        iterable = tqdm(self.__range) if (
            self.verbose and self.n_jobs == 1) else self.__range
        if self.n_jobs == 1:
            for n_clusters in iterable:
                cost = self._elbow_method_for_content(n_clusters)
                self.__costs.append(cost)
        else:
            self.__costs = Parallel(n_jobs=self.n_jobs, verbose=False)(
                delayed(self._elbow_method_for_content)(n_clusters) for n_clusters in iterable
            )

        self.optimal_clusters_ = self.__range[self.__compute_elbow_point(
            self.__costs)]
        self.estimator.n_clusters = self.optimal_clusters_

    def __create_dm__kmodes(self, dataset):
        if self.verbose:
            print("Creating distance matrix...")
        if type(dataset).__name__ == "DataFrame":
            dataset = dataset.values
        lenDataset = len(dataset)
        distance_matrix = np.zeros(lenDataset * lenDataset).reshape(
            lenDataset, lenDataset
        )
        for i in range(lenDataset):
            for j in range(lenDataset):
                x1 = dataset[i].reshape(1, -1)
                x2 = dataset[j].reshape(1, -1)
                distance = matching_dissim(x1, x2)
                distance_matrix[i][j] = distance[0]
                distance_matrix[j][i] = distance[0]
        return distance_matrix

    def __mixed_distance__kproto(self, a, b, alpha=0.01):
        if self.__kwargs[self.__cat_name[0]] is None:
            num_score = euclidean_dissim(a, b)
            return num_score
        else:
            cat_index = self.__kwargs[self.__cat_name[0]]
            a_cat = []
            b_cat = []
            for index in cat_index:
                a_cat.append(a[index])
                b_cat.append(b[index])
            a_num = []
            b_num = []
            l = len(a)
            for index in range(l):
                if index not in cat_index:
                    a_num.append(a[index])
                    b_num.append(b[index])

            a_cat = np.array(a_cat).reshape(1, -1)
            a_num = np.array(a_num).reshape(1, -1)
            b_cat = np.array(b_cat).reshape(1, -1)
            b_num = np.array(b_num).reshape(1, -1)
            cat_score = matching_dissim(a_cat, b_cat)
            num_score = euclidean_dissim(a_num, b_num)
            return cat_score + num_score * alpha

    def __create_dm__kproto(self, dataset, alpha=0.1):
        if type(dataset).__name__ == "DataFrame":
            dataset = dataset.values
        lenDataset = len(dataset)
        if self.verbose:
            print("Creating distance matrix...")
        distance_matrix = np.zeros(lenDataset * lenDataset).reshape(
            lenDataset, lenDataset
        )
        for i in range(lenDataset):
            for j in range(lenDataset):
                x1 = dataset[i]
                x2 = dataset[j]
                distance = self.__mixed_distance__kproto(x1, x2, alpha=alpha)
                distance_matrix[i, j] = distance[0]
                distance_matrix[j, i] = distance[0]
        return distance_matrix

    def _silhuette_method_for_kmeans_kmedoids(self, n_clusters):
        if self.__is_pipeline:
            self.estimator[-1].n_clusters = n_clusters
            self.estimator.fit(self.__X, **self.__kwargs)
            return silhouette_score(self.__X, self.estimator[-1].labels_)
        else:
            self.estimator.n_clusters = n_clusters
            self.estimator.fit(self.__X, **self.__kwargs)
            return silhouette_score(self.__X, self.estimator.labels_)
        
    def _silhuette_method_for_kmodes_kproto(self, n_clusters, distance_matrix):
        if self.__is_pipeline:
            self.estimator[-1].n_clusters = n_clusters
            self.estimator.fit(self.__X, **self.__kwargs)
            cluster_labels = self.estimator[-1].labels_
        else:
            self.estimator.n_clusters = n_clusters
            self.estimator.fit(self.__X, **self.__kwargs)
            cluster_labels = self.estimator.labels_
        
        return silhouette_score(
            distance_matrix, cluster_labels, metric="precomputed"
        )

    def __silhuette_method(self):
        self.__sils = []
        iterable = tqdm(self.__range) if (self.verbose and self.n_jobs == 1) else self.__range
        if self.verbose:
            print("Calculating silhouette scores...")
        if self.__model_name in ["KMeans", "KMedoids"]:
            if self.n_jobs == 1:
                for n_clusters in iterable:
                    sil = self._silhuette_method_for_kmeans_kmedoids(n_clusters)
                    self.__sils.append(sil)
            else:
                self.__sils = Parallel(n_jobs=self.n_jobs, verbose=False)(
                    delayed(self._silhuette_method_for_kmeans_kmedoids)(n_clusters) for n_clusters in iterable
                )
        elif self.__model_name == "KModes":
            distance_matrix = self.__create_dm__kmodes(self.__X)
            if self.n_jobs == 1:
                for n_clusters in iterable:
                    score = self._silhuette_method_for_kmodes_kproto(n_clusters, distance_matrix)
                    self.__sils.append(score)
            else:
                self.__sils = Parallel(n_jobs=self.n_jobs, verbose=False)(
                    delayed(self._silhuette_method_for_kmodes_kproto)(n_clusters, distance_matrix) for n_clusters in iterable
                )
        elif self.__model_name == "KPrototypes":
            self.__cat_name = [k for k in self.__kwargs.keys() if k.endswith("categorical")]
            if len(self.__cat_name) == 0:
                raise Exception(
                    "The argument 'categorical' is required for the KPrototypes estimator."
                )
            if not isinstance(self.__kwargs[self.__cat_name[0]], list):
                raise Exception("The argument 'categorical' must be a list.")
            if not self.__kwargs.get("alpha"):
                self.__alpha = 0.01
            else:
                self.__alpha = self.__kwargs["alpha"]
                del self.__kwargs["alpha"]
            distance_matrix = self.__create_dm__kproto(
                self.__X, alpha=self.__alpha)
            if self.n_jobs == 1:
                for n_clusters in iterable:
                    score = self._silhuette_method_for_kmodes_kproto(n_clusters, distance_matrix)
                    self.__sils.append(score)
            else:
                self.__sils = Parallel(n_jobs=self.n_jobs, verbose=False)(
                    delayed(self._silhuette_method_for_kmodes_kproto)(n_clusters, distance_matrix) for n_clusters in iterable
                )
        else:
            raise Exception(
                f"The estimator '{self.__model_name}' is not implemented yet."
            )

        self.optimal_clusters_ = self.__range[np.argmax(self.__sils)]
        self.estimator.n_clusters = self.optimal_clusters_

    def __plot_elbow(self, ax):
        if ax is None:
            plt.figure(figsize=(10, 6))
            plt.plot(self.__range, self.__costs, "-o")
            plt.xlabel("Number of clusters")
            plt.ylabel("Cost")
            plt.title("Elbow Method For Optimal Cluster Number")
            plt.axvline(
                x=self.optimal_clusters_,
                color="r",
                linestyle="--",
                label=f"Elbow point = {self.optimal_clusters_}",
            )
            plt.legend()
        else:
            ax.plot(self.__range, self.__costs, "-o")
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel("Cost")
            ax.set_title("Elbow Method For Optimal Cluster Number")
            ax.axvline(
                x=self.optimal_clusters_,
                color="r",
                linestyle="--",
                label=f"Elbow point = {self.optimal_clusters_}",
            )
            ax.legend()

    def __plot_silhouette(self, ax):
        if ax is None:
            plt.figure(figsize=(10, 6))
            plt.plot(self.__range, self.__sils, "-o")
            plt.xlabel("Number of clusters")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Method For Optimal Cluster Number")
            plt.axvline(
                x=self.optimal_clusters_,
                color="r",
                linestyle="--",
                label=f"Maximum silhouette point = {self.optimal_clusters_}",
            )
            plt.legend()
        else:
            ax.plot(self.__range, self.__sils, "-o")
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Method For Optimal Cluster Number")
            ax.axvline(
                x=self.optimal_clusters_,
                color="r",
                linestyle="--",
                label=f"Maximum silhouette point = {self.optimal_clusters_}",
            )
            ax.legend()

    def plot(self, *, ax=None):
        """
Plots the results of the selected method (either 'elbow' or 'silhouette') to visualize the optimal cluster number.

Parameters:
-----------
ax : matplotlib axis object, optional
    Axis on which to draw the plot. If None, a new figure and axis will be created.

Returns:
--------
None

Example:
--------

>>> from sklearn.cluster import KMeans
>>> from sklean.datasets import load_iris
>>> data = load_iris().data
>>> model = NClusterSearch(estimator=KMeans(), method='elbow')
>>> model.fit(data)
>>> model.plot()
        """
        if not self.__fitted:
            raise Exception(
                "The model is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
        if ax is not None and not isinstance(ax, plt.Axes):
            raise ValueError(
                "The argument 'ax' must be a matplotlib Axes object.")
        if self.method == "elbow":
            self.__plot_elbow(ax=ax)
        elif self.method == "silhouette":
            self.__plot_silhouette(ax=ax)

    def predict(self, X, **kwargs):
        """
Predicts the closest cluster for each sample in `X` using the `best_estimator_`.

Parameters:
-----------
`X` : array-like, shape (n_samples, n_features)
    New data to predict cluster labels.

`**kwargs` : Additional keyword arguments to be passed to the estimator's predict method.

Returns:
--------
`labels` : array, shape (n_samples,)
    Index of the cluster each sample belongs to.

Example:
--------

>>> from sklearn.cluster import KMeans
>>> from sklean.datasets import load_iris
>>> data = load_iris().data
>>> model = NClusterSearch(estimator=KMeans(), method='elbow')
>>> model.fit(data)
>>> predictions = model.predict(new_data)
        """
        if not self.__fitted:
            raise Exception(
                "The model is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        return self.best_estimator_.predict(X, **kwargs)

    @property
    def labels_(self):
        """
Returns the labels of each point for the best estimator.

This property can be accessed only after calling the fit method.

Returns:
--------
labels : array, shape (n_samples,)
    Index of the cluster each sample belongs to.

Example:
--------

>>> from sklearn.cluster import KMeans
>>> from sklean.datasets import load_iris
>>> data = load_iris().data
>>> model = NClusterSearch(estimator=KMeans(), method='elbow')
>>> model.fit(data)
>>> labels = model.labels_
        """
        return self.best_estimator_.labels_

    def __repr__(self) -> str:
        return f"NClusterSearch(estimator={self.estimator}, method='{self.method}', min_clusters={self.min_clusters}, max_clusters={self.max_clusters}, step={self.step}, random_state={self.random_state})"
