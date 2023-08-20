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

   .. method:: __init__(X, y, penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio)

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

      Returns the deviance of the model.

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



Stepwise Selection for Linear Models
------------------------------------

.. class:: Stepwise(formula, data, model, direction, criterion, alpha, max_iter, formula_params, fit_params, verbose)


   The `Stepwise` class provides a method to perform stepwise model selection, which is a method to add or remove predictors based on their significance, AIC or BIC in a model.

   :param formula: A string representing the formula, using the `patsy` formula syntax.
   :type formula: str
   :param data: A pandas DataFrame that contains the data for both the dependent and independent variables.
   :type data: DataFrame
   :param model: Specifies the type of model to be used.
   :type model: Union[GLM, OLS, Logit, LogisticRegression]
   :param direction: Specifies the direction of the stepwise process.
   :type direction: Literal["both", "forward", "backward"]
   :param criterion: The criterion to be used for adding or removing predictors.
   :type criterion: Literal["aic", "bic", "f-test"]
   :param alpha: The significance level for adding or removing predictors. It must be a value between 0 and 1.
   :type alpha: float
   :param max_iter: The maximum number of iterations for the both direction process.
   :type max_iter: int
   :param formula_params: Additional parameters to be passed to the model's `from_formula` method.
   :type formula_params: Dict[str, Any]
   :param fit_params: Additional parameters to be passed to the model's `fit` method.
   :type fit_params: Dict[str, Any]
   :param verbose: If set to `False`, the class will not print information about the stepwise process.
   :type verbose: bool

   .. attribute:: optimal_model_

      The optimal model obtained after the stepwise process.

   .. attribute:: optimal_formula_

      The optimal model formula after the stepwise process.

   .. attribute:: optimal_variables_

      List of optimal predictor variables in the final model.

   .. attribute:: optimal_metric_

      The optimal value of the chosen criterion (e.g., AIC, BIC, or F-test) for the final model.

   .. method:: fit()

      Conducts the stepwise process based on the specified direction and criterion.

      **Examples**:

      .. jupyter-execute::
         :hide-code:

         %config InlineBackend.figure_format = 'retina'

      .. jupyter-execute::

         import pandas as pd
         from statsmodels.api import OLS
         from estyp.linear_model import Stepwise
         data = pd.DataFrame({"y": [1,2,3,4,5], "x1": [5,20,3,2,1], "x2": [6,7,8,9,10]})
         stepwise = Stepwise(formula="y ~ 1", data=data, model=OLS, direction="forward", criterion="aic")
         stepwise.fit()
         print("Best predictors:", stepwise.optimal_variables_)

   .. method:: plot_history(ax=None)

      Plots the history of the chosen criterion during the stepwise.

      :param ax: An Axes instance for the plot. If not provided, a new figure and axes will be created.
      :type ax: matplotlib.axes.Axes, optional

      **Returns**:

      fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
         The Figure and Axes instances containing the plot if not provided.

      **Examples**:

      .. jupyter-execute::

         import pandas as pd
         from statsmodels.api import OLS
         from estyp.linear_model import Stepwise

         data = pd.DataFrame(
            {
               "y": [1, 2, 3, 4, 5],
               "x1": [5, 20, 3, 2, 1],
               "x2": [6, 7, 8, 9, 10],
               "x3": [1, 2, 40, 4, 30],
               "x4": [20, 1, 4, 5, 6],
               "x5": [90, -1, 40, 5, 26],
            }
         )
         stepwise = Stepwise(
            formula="y ~ x1 + x2 + x3 + x4 + x5",
            data=data,
            model=OLS,
            direction="backward",
            criterion="bic",
         )
         stepwise.fit()
         fig, ax = stepwise.plot_history()

   
   Example
   -------

   .. jupyter-execute::

      import pandas as pd
      from statsmodels.api import OLS
      from estyp.linear_model import Stepwise
      data = pd.DataFrame({"y": [1,2,3,4,5], "x1": [5,20,3,2,1], "x2": [6,7,8,9,10]})
      stepwise = Stepwise(formula="y ~ 1", data=data, model=OLS, direction="forward", criterion="aic")
      stepwise.fit()

.. note::

   - The class is designed to work seamlessly with statsmodels models.
   - If using "both" as the direction, the "f-test" criterion is not available.
   - Ensure that the data provided is appropriate for the model chosen.
