The `linear\_model.stepwise` module
====================================

Both Method Variable Selection
------------------------------

.. function:: both_selection(formula, data, model, max_iter, formula_kwargs, fit_kwargs)

    Both Forward and Backward Variable Selection for GLM's
    ------------------------------------------------------

    This function performs both forward and backward variable selection using the Akaike Information Criterion (AIC).

   :param formula: A string representing the initial model formula.
   :type formula: str
   :param data: A Pandas DataFrame containing the data to be used for model fitting.
   :type data: DataFrame
   :param model: A statsmodels.GLM object that represents the type of model to be fit.
   :type model: GLM, OLS, Logit, LogisticRegression
   :param max_iter: The maximum number of iterations to perform.
   :type max_iter: int
   :param formula_kwargs: Additional keyword arguments to be passed to the model.from_formula() method.
   :type formula_kwargs: dict
   :param fit_kwargs: Additional keyword arguments to be passed to the fit() method. Defaults to a dictionary :code:`{"disp"\:0}`.


   :returns: A string representing the final model formula.

   .. jupyter-execute::

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

Forward Variable Selection
--------------------------

.. function:: forward_selection(y, data, model, alpha, formula_kwargs, fit_kwargs)

    Forward Variable Selection for GLM's
    ------------------------------------
    
    This function performs forward variable selection using p-values calculated from nested models testing.
  
   :param y: A string containing the name of the dependent variable (target) to be predicted.
   :type y: str
   :param data: The pandas DataFrame containing both the target variable 'y' and the predictor variables for model training.
   :type data: DataFrame
   :param model: A statsmodels model class. The statistical model to be used for model fitting and evaluation. Defaults to :code:`sm.OLS`.
   :type model: Union[GLM, OLS, Logit, LogisticRegression]
   :param alpha: A number between 0 and 1. The significance level for feature selection. A feature is added to the model if its p-value is less than this alpha value. Defaults to 0.05.
   :type alpha: float
   :param formula_kwargs: Additional keyword arguments to be passed to the model.from_formula() method. Defaults to :code:`dict()`.
   :type formula_kwargs: dict
   :param fit_kwargs: Additional keyword arguments to be passed to the fit() method. Defaults to a dictionary :code:`{"disp"\:0}`.
   :type fit_kwargs: dict

   :returns: A string representing the final model formula.

   .. jupyter-execute::

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

