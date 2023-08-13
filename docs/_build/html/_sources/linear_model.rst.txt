The `linear\_model` module
===========================


.. toctree::
   :maxdepth: 2

   linear_model.stepwise


Module contents
---------------

Logistic Regression
-------------------

.. class:: LogisticRegression(X, y, penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio)


   This class implements a logistic regression model. It inherits from the `sklearn.linear_model.LogisticRegression` class, but adds additional methods for calculating confidence intervals, p-values, and model summaries.

   .. method:: __init__(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio)

      :param X: A Pandas DataFrame or a NumPy array containing the model predictors.
      :type X: Union[DataFrame, ndarray, None]
      :param y: A Pandas Series or a NumPy array containing the model response.
      :type y: Union[Series, ndarray, None]
      :param penalty: The type of penalty to use. Can be one of ``"none"`` (default). ``"l1"``, ``"l2"``, or ``"elasticnet"``.
      :type penalty: Literal['l1', 'l2', 'elasticnet']
      :param dual: Whether to use the dual formulation of the problem.
      :type dual: bool
      :param tol: The tolerance for convergence.
      :type tol: float
      :param C: The regularization strength.
      :type C: int
      :param fit_intercept: Whether to fit an intercept term.
      :type fit_intercept: bool
      :param intercept_scaling: The scaling factor for the intercept term.
      :type intercept_scaling: int
      :param class_weight: None (default), "balanced" or a dictionary that maps class labels to weights.
      :type class_weight: Union[None, str, dict]
      :param random_state: The random seed.
      :type random_state: int
      :param solver: The solver to use. Can be one of ``"lbfgs"`` (default), ``"liblinear"``, ``"newton-cg"``, ``"newton-cholesky"``, ``"sag"``, or ``"saga"``.
      :type solver: Literal['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
      :param max_iter: The maximum number of iterations.
      :type max_iter: int
      :param multi_class: The type of multi-class classification to use. Can be one of ``"auto"``, ``"ovr"``, or ``"multinomial"``.
      :type multi_class: Literal['auto', 'ovr', 'multinomial']
      :param verbose: The verbosity level.
      :type verbose: int
      :param warm_start: Whether to use the warm start.
      :type warm_start: bool
      :param n_jobs: The number of jobs to use for parallel processing.
      :type n_jobs: int
      :param l1_ratio: The l1_ratio parameter for elasticnet regularization.
      :type l1_ratio: Union[float, None]



   .. method:: fit()

      Fits the model to the data.

   .. method:: predict(new_data: DataFrame)

      Predicts the class labels for new data.

   .. method:: conf_int(conf_level=0.95)

      Calculates the confidence intervals for the model coefficients.

   .. method:: se()

      Calculates the standard errors for the model coefficients.

   .. method:: z_values()

      Calculates the z-scores for the model coefficients.

   .. method:: p_values()

      Calculates the p-values for the model coefficients.

   .. method:: summary(conf_level=0.95)

      Prints a summary of the model.

   .. method:: from_formula(formula, data)

      Class method to create an instance from a formula.

   .. attribute:: params

      Returns the estimated values for model parameters.

   .. attribute:: aic

      Calculates the Akaike information criterion (AIC) for the model.

   .. attribute:: bic

      Calculates the Bayesian information criterion (BIC) for the model.

   .. attribute:: cov_matrix

      Returns the estimated covariance matrix for model parameters.

   .. attribute:: residuals

      Returns the deviance of the models

   .. attribute:: deviance_residuals

      Returns the deviance residuals.



   Examples
   --------

   .. jupyter-execute::

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
