Getting Started
===============

**First**: `Install <./instalation.html>`_ ESTYP library if you haven't already:

Here are some examples of how to use the library:

Model Selection
---------------

We will now select a logistic regression model that best classifies the versicolor category.

First, we load the data:

.. jupyter-execute::

    from sklearn.datasets import load_iris
    import pandas as pd

    content = load_iris()

    data = pd.DataFrame(content.data, columns=[f"x{i+1}" for i in range(content.data.shape[1])])
    data["y"] = (content.target == 1).astype(int)
    print(data.head())

Then, we run a model selection process with forward and both (forward and backward) steps:

Review `LogisticRegression() <./linear_model.html#LogisticRegression>`_, `forward_selection() <./linear_model.stepwise.html#forward-variable-selection>`_ and `both_selection() <./linear_model.stepwise.html#both-method-variable-selection>`_ documentation for more information and other parameters.

.. jupyter-execute::

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

Now we choose between the two resultant models using nested models test:

View `nested_models_test() <./testing.html#nested-models-f-test-function>`_ documentation for more information and other parameters.

.. jupyter-execute::

    from estyp.testing import nested_models_test

    model1 = LogisticRegression.from_formula(ff1, data).fit()
    model2 = LogisticRegression.from_formula(ff2, data).fit()

    nested_models_test(model1, model2) # First model is nested in the second one

With :math:`\alpha=0.05`, the null hypothesis is not rejected: model2 is significantly not better than model1.


Means equality between two samples
----------------------------------

We will now test if the mean of the x1 and x4 columns are equal.

View the details of the `t-test <https://en.wikipedia.org/wiki/Student%27s_t-test>`_ for more information.

Review `t_test() <./testing.html#student-s-t-test>`_ documentation for more information and other parameters.

.. jupyter-execute::

    from estyp.testing import t_test

    x = data["x1"]
    y = data["x4"]

    test_result = t_test(x, y)
    print(test_result)

With :math:`\alpha=0.05`, the null hypothesis is rejected: mean of x is significantly different from the mean of y.

Equality in variances of two samples
------------------------------------

We will now test if the variance of the x1 and x4 columns are equal.

View the details of the `variance test <https://en.wikipedia.org/wiki/F-test_of_equality_of_variances>`_ for more information.

Review `var_test() <./testing.html#f-test-to-compare-two-variances>`_ documentation for more information and other parameters.

.. jupyter-execute::

    from estyp.testing import var_test

    test_result = var_test(x, y)
    print(test_result)

With :math:`\alpha=0.05`, the null hypothesis is not rejected: variance of x is significantly equal from the variance of y.

Correlation between two samples
-------------------------------

We will now test if the correlation between x1 and x4 is greater than 0.

Review `cor_test() <./testing.html#test-for-association-correlation-between-paired-samples>`_ documentation for more information and other parameters.

.. jupyter-execute::

    from estyp.testing import cor_test

    test_result = cor_test(x, y, alternative="greater", method="spearman")
    print(test_result)

With :math:`\alpha=0.05`, the null hypothesis is rejected: Spearman correlation between x and y is significantly greater than 0.

Proportions testing
-------------------

We will now test if the proportion of non versicolor flowers is equal to 0.75.

Review `prop_test() <./testing.html#test-of-equal-or-given-proportions>`_ documentation for more information and other parameters.

.. jupyter-execute::

    from estyp.testing import prop_test

    counts = data["y"].value_counts()

    test_result = prop_test(counts, p=0.75)
    print(test_result)

With :math:`\alpha=0.05`, the null hypothesis is rejected: proportion of non versicolor flowers is not 0.75.

Searching Optimal Number of Clusters 
------------------------------------

We will now search for the optimal number of clusters in the iris dataset, powered by the `elbow method <https://en.wikipedia.org/wiki/Elbow_method_(clustering)>`_.

Review `NClusterSearch() <./cluster.html#optimal-number-of-clusters-searcher>`_ documentation for more information and other parameters.

.. jupyter-execute::
    :hide-code:

    %config InlineBackend.figure_format = 'retina'

.. jupyter-execute::

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

The number of clusters suggested is 3.

Linear Regression Model Assumptions
-----------------------------------

We will now test the assumptions of a linear regression model.

Review `CheckModel() <./testing.html#checkmodel-class>`_ documentation for more information and other parameters.

.. jupyter-execute::

    from estyp.testing import CheckModel
    import statsmodels.api as sm

    model = sm.OLS.from_formula('x4 ~ x1 + x2 + x3', data=data).fit()
    checker = CheckModel(model)
    checker.check_all()

Apparently we only approve the residuals normality assumption.