from typing import Literal

import numpy as np
from pandas import DataFrame, Series, concat
from patsy import build_design_matrices, dmatrices
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression as LogisticRegression_


class LogisticRegression(LogisticRegression_):
    """
Logistic Regression
-------------------

Description
-----------

This class implements a logistic regression model. It inherits from the `sklearn.linear_model.LogisticRegression` class, but adds additional methods for calculating confidence intervals, p-values, and model summaries.

Parameters
----------

- `penalty`: The type of penalty to use. Can be one of `"none"` (default). `"l1"`, `"l2"`, or `"elasticnet"`.
- `dual`: Whether to use the dual formulation of the problem.
- `tol`: The tolerance for convergence.
- `C`: The regularization strength.
- `fit_intercept`: Whether to fit an intercept term.
- `intercept_scaling`: The scaling factor for the intercept term.
- `class_weight`: None (default), "balanced" or a dictionary that maps class labels to weights.
- `random_state`: The random seed.
- `solver`: The solver to use. Can be one of `"lbfgs"` (default), `"liblinear"`, `"newton-cg"`, `"newton-cholesky"`, `"sag"`, or `"saga"`.
- `max_iter`: The maximum number of iterations.
- `multi_class`: The type of multi-class classification to use. Can be one of `"auto"`, `"ovr"`, or `"multinomial"`.
- `verbose`: The verbosity level.
- `warm_start`: Whether to use the warm start.
- `n_jobs`: The number of jobs to use for parallel processing.
- `l1_ratio`: The l1_ratio parameter for elasticnet regularization.

Properties
----------

- `params`: Returns the estimated values for model parameters.
- `aic`: Calculates the Akaike information criterion (AIC) for the model.
- `bic`: Calculates the Bayesian information criterion (BIC) for the model.
- `cov_matrix`: Returns the covariance matrix for model parámetres.

Methods
-------

- `fit()`: Fits the model to the data.
- `predict()`: Predicts the class labels for new data.
- `conf_int()`: Calculates the confidence intervals for the model coefficients.
- `se()`: Calculates the standard errors for the model coefficients.
- `z_values()`: Calculates the z-scores for the model coefficients.
- `p_values()`: Calculates the p-values for the model coefficients.
- `summary()`: Prints a summary of the model.

Examples
--------

>>> import numpy as np
>>> import pandas as pd
>>> np.random.seed(123)
>>> data = pd.DataFrame({
...     "y": np.random.randint(2, size=100),
...     "x1": np.random.uniform(-1, 1, size=100),
...     "x2": np.random.uniform(-1, 1, size=100),
... })
>>> formula = "y ~ x1 + x2"
>>> spec = LogisticRegression.from_formula(formula, data)
>>> model = spec.fit()
>>> print(model.summary())
    """
    
    def __init__(self, penalty: Literal['l1', 'l2', 'elasticnet'] = None, *, dual = False, tol = 0.0001, C = 1, fit_intercept = False, intercept_scaling = 1, class_weight = None, random_state = 2023, solver: Literal['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'] = "lbfgs", max_iter = 100, multi_class: Literal['auto', 'ovr', 'multinomial'] = "auto", verbose = 0, warm_start: bool = False, n_jobs = -1, l1_ratio = None):
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)

    @classmethod
    def from_formula(cls, formula, data):
        y, X = dmatrices(formula, data)
        cls.y = y.reshape(-1)
        cls.formula = formula
        cls.X = X
        return cls()

    def __log_likelihood(self):
        z = np.dot(self.X, self.coef_.flatten())
        log_likelihood = np.sum(self.y * z - np.log(1 + np.exp(z)))
        return log_likelihood


    def fit(self):
        super().fit(self.X, self.y)
        return self
    
    def conf_int(self, conf_level = 0.95):
        z_score = norm.ppf(1 - (1 - conf_level) / 2)
        standard_errors = np.sqrt(np.diag(self.cov_matrix))
        lower_limits = self.coef_[0] - z_score * standard_errors
        upper_limits = self.coef_[0] + z_score * standard_errors
        ci = DataFrame({
            "[Lower,": lower_limits,
            "Upper]": upper_limits
        }, index = self.X.design_info.column_names)
        return ci
    
    def se(self):
        return Series(np.sqrt(np.diag(self.cov_matrix)), index=self.X.design_info.column_names, name="S.E.")

    def z_values(self):
        z_est = self.coef_[0] / self.se()
        return Series(z_est, index=self.X.design_info.column_names, name="z")
    
    def p_values(self):
        p_values = 2 * norm.cdf(-np.abs(self.z_values()))
        return Series(p_values, index=self.X.design_info.column_names, name="Pr(>|z|)")
    
    def summary(self, conf_level = 0.95):
        summary = concat(
            objs = [self.params, self.se(), self.z_values(), self.p_values(), self.conf_int(conf_level)], 
            axis = 1
        )
        return summary
    
    def predict(self, new_data: DataFrame):
        dsg = build_design_matrices([self.X.design_info], new_data)[0].view()
        return super().predict_proba(dsg)[:, 1]

        
    @property
    def cov_matrix(self):
        p = self.predict_proba(self.X)[:, 1]
        hess_matrix = np.dot(self.X.T, np.dot(np.diag(p * (1 - p)), self.X))
        cov_matrix = np.linalg.inv(hess_matrix)
        return cov_matrix
    

    @property
    def params(self):
        return Series(self.coef_.flatten().tolist(), index=self.X.design_info.column_names, name="Estimate")
    
    @property
    def aic(self):
        aic = -2 * self.__log_likelihood() + 2 * self.coef_.shape[0] - 1
        return aic.item()
    
    @property
    def bic(self):
        bic = -2 * self.__log_likelihood() + np.log(self.X.shape[0]) * (self.coef_.shape[0] - 1)
        return bic.item()
    
    
    def __repr__(self) -> str:
        return "LogisticRegression()"