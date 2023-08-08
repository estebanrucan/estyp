Getting Started
===============

**First**: `Install <./instalation.html>`_ ESTYP library if you haven't already:

Here are some examples of how to use the library:

Model Selection
---------------

We will now select a logistic regression model that best classifies the versicolor category.

Review `LogisticRegression <>`_, `forward_selection <>`_ and `both_selection <>`_ documentation for more information.

First, we load the data:

.. jupyter-execute::

    from sklearn.datasets import load_iris
    import pandas as pd

    content = load_iris()

    data = pd.DataFrame(content.data, columns=[f"x{i+1}" for i in range(content.data.shape[1])])
    data["y"] = (content.target == 1).astype(int)
    print(data.head())

Then, we run a model selection process with forward and both (forward and backward) steps:

.. jupyter-execute::

    from estyp.linear_model.stepwise import forward_selection, both_selection
    from estyp.linear_model import LogisticRegression

    ff1 = forward_selection(
        y       = "y",
        data    = data,
        model   = LogisticRegression,
        verbose = False,
    )
    ff2 = both_selection(
        formula = formula1,
        data    = data,
        model   = LogisticRegression,
        verbose = False
    )
    print("- Forward result:", ff1)
    print("- Both result   :", ff2)
