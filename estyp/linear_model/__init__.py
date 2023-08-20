from typing import Any, Dict, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series, concat
from patsy import build_design_matrices, dmatrices
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression as LogisticRegression_
from statsmodels.api import GLM, OLS, Logit

from estyp.testing.__base import __nested_models_test as nested_models_test
from estyp.testing.__base import bcolors


class LogisticRegression(LogisticRegression_):
    """
    Logistic Regression
    ===================

    View this in the [online documentation](https://estyp.readthedocs.io/en/latest/linear_model.html#logistic-regression).

    Description
    -----------

    This class implements a logistic regression model. It inherits from the `sklearn.linear_model.LogisticRegression` class, but adds additional methods for calculating confidence intervals, p-values, and model summaries.

    Parameters
    ----------

    - `X`: A Pandas DataFrame or a NumPy array containing the model predictors.
    - `y`: A Pandas Series or a NumPy array containing the model response.
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
    - `cov_matrix`: Returns the covariance matrix for model parametres.
    - `deviance`: Returns the deviance of the model.
    - `deviance_residuals`: Returns the deviance residuals.

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

    def __init__(
        self,
        X: Union[DataFrame, np.ndarray] = None,
        y: Union[Series, np.ndarray] = None,
        penalty: Literal["l1", "l2", "elasticnet"] = None,
        *,
        dual=False,
        tol=0.0001,
        C=1,
        fit_intercept=False,
        intercept_scaling=1,
        class_weight=None,
        random_state=2023,
        solver: Literal[
            "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
        ] = "lbfgs",
        max_iter=100,
        multi_class: Literal["auto", "ovr", "multinomial"] = "auto",
        verbose=0,
        warm_start: bool = False,
        n_jobs=-1,
        l1_ratio=None,
    ):
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )

        if self.__dict__.get("X") is None and self.__dict__.get("y") is None:
            if isinstance(X, np.ndarray):
                X = DataFrame(X, columns=[f"var_{i+1}" for i in range(X.shape[1])])
            if X is not None and y is not None:
                data = concat([X, Series(y, name="y")], axis=1)
                formula = (
                    f"y ~ {' + '.join(X.columns.tolist()) if X.columns.tolist() else 1}"
                )
                y, X = dmatrices(formula, data)
                self.X = X
                self.y = y.reshape(-1)

    @classmethod
    def from_formula(cls, formula, data, **kwargs):
        if "X" in kwargs.keys() or "y" in kwargs.keys():
            raise ValueError(
                "The 'X' and 'y' arguments cannot be used with the 'from_formula' method."
            )
        y, X = dmatrices(formula, data)
        self_ = cls(**kwargs)
        self_.y = y.reshape(-1)
        self_.X = X
        self_.formula = formula
        return self_

    def __log_likelihood(self):
        z = np.dot(self.X, self.coef_.flatten())
        log_likelihood = np.sum(self.y * z - np.log(1 + np.exp(z)))
        return log_likelihood

    def fit(self, disp=None):
        super().fit(self.X, self.y)
        return self

    def conf_int(self, conf_level=0.95):
        z_score = norm.ppf(1 - (1 - conf_level) / 2)
        standard_errors = np.sqrt(np.diag(self.cov_matrix))
        lower_limits = self.coef_[0] - z_score * standard_errors
        upper_limits = self.coef_[0] + z_score * standard_errors
        ci = DataFrame(
            {"[Lower,": lower_limits, "Upper]": upper_limits},
            index=self.X.design_info.column_names,
        )
        return ci

    def se(self):
        return Series(
            np.sqrt(np.diag(self.cov_matrix)),
            index=self.X.design_info.column_names,
            name="S.E.",
        )

    def z_values(self):
        z_est = self.coef_[0] / self.se()
        return Series(z_est, index=self.X.design_info.column_names, name="z")

    def p_values(self):
        p_values = 2 * norm.cdf(-np.abs(self.z_values()))
        return Series(p_values, index=self.X.design_info.column_names, name="Pr(>|z|)")

    def summary(self, conf_level=0.95):
        summary = concat(
            objs=[
                self.params,
                self.se(),
                self.z_values(),
                self.p_values(),
                self.conf_int(conf_level),
            ],
            axis=1,
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
        return Series(
            self.coef_.flatten().tolist(),
            index=self.X.design_info.column_names,
            name="Estimate",
        )

    @property
    def aic(self):
        aic = -2 * self.__log_likelihood() + 2 * (self.coef_.shape[0] - 1)
        return aic.item()

    @property
    def bic(self):
        bic = -2 * self.__log_likelihood() + np.log(self.X.shape[0]) * (
            self.coef_.shape[0] - 1
        )
        return bic.item()

    @property
    def deviance_residuals(self):
        p = self.predict_proba(self.X)[:, 1]
        residuals = -2 * (self.y * np.log(p) + (1 - self.y) * np.log(1 - p))
        return residuals

    @property
    def deviance(self):
        return np.sum(self.deviance_residuals).item()

    def __repr__(self) -> str:
        return "LogisticRegression()"


class Stepwise:
    """
    Stepwise Selection for Linear Models
    ======================================
    
    View this in the [online documentation](https://estyp.readthedocs.io/en/latest/linear_model.html#stepwise-selection-for-linear-models).
    
    Description
    -----------

    The `Stepwise` class provides a method to perform stepwise model selection, which is a method to add or remove predictors based on their significance, AIC or BIC in a model.

    Parameters:
    -----------
    `formula` : str
        A string representing the formula, using the `patsy` formula syntax.
        For instance, the formula "y ~ x1 + x2" denotes `y` as the dependent variable and `x1` and `x2` as independent variables.

    `data` : DataFrame
        A pandas DataFrame that contains the data for both the dependent and independent variables.

    `model` : Union[GLM, OLS, Logit, LogisticRegression], optional (default = OLS)
        Specifies the type of model to be used. The options are:
        - `GLM`: Generalized Linear Model
        - `OLS`: Ordinary Least Squares Regression
        - `Logit`: Logistic Regression (for binary classification)
        - `LogisticRegression`: Logistic Regression (from the `estyp.linear_model` module)

    `direction` : Literal["both", "forward", "backward"], optional (default = "forward")
        Specifies the direction of the stepwise process:
        - "forward": Start with no predictors and add predictors one-by-one.
        - "backward": Start with all predictors and remove predictors one-by-one.
        - "both": Combination of forward and backward methods.

    `criterion` : Literal["aic", "bic", "f-test"], optional (default = "aic")
        The criterion to be used for adding or removing predictors:
        - "aic": Akaike Information Criterion
        - "bic": Bayesian Information Criterion
        - "f-test": F-test (only available for "forward" and "backward" directions)

    `alpha` : float, optional (default = 0.05)
        The significance level for adding or removing predictors.
        It must be a value between 0 and 1.

    `max_iter` : int, optional (default = 100)
        The maximum number of iterations for the both direction process.

    `formula_params` : Dict[str, Any], optional (default = {})
        Additional parameters to be passed to the model's `from_formula` method.

    `fit_params` : Dict[str, Any], optional (default = {"disp": 0})
        Additional parameters to be passed to the model's `fit` method.

    `verbose` : bool, optional (default = True)
        If set to `False`, the class will not print information about the stepwise process.

    Attributes:
    -----------
    `optimal_model_`  : Model instance
        The optimal model obtained after the stepwise process.

    `optimal_formula_` : str
        The optimal model formula after the stepwise process.

    `optimal_variables_` : list
        List of optimal predictor variables in the final model.

    `optimal_metric_` : float
        The optimal value of the chosen criterion (e.g., AIC, BIC, or F-test) for the final model.

    Methods:
    --------
    `fit()` :
        Conducts the stepwise process based on the specified direction and criterion.

    `plot_history(ax=None)`:
        Plots the history of the chosen criterion during the stepwise.

        Parameters:
        - `ax  : matplotlib.axes.Axes (optional)
            An Axes instance for the plot. If not provided, a new figure and axes will be created.

    Example:
    ---------
    >>> import pandas as pd
    >>> from statsmodels.api import OLS
    >>> from estyp.linear_model import Stepwise
    >>> data = pd.DataFrame({"y": [1,2,3,4,5], "x1": [5,20,3,2,1], "x2": [6,7,8,9,10]})
    >>> stepwise = Stepwise(formula="y ~ 1", data=data, model=OLS, direction="forward", criterion="aic")
    >>> stepwise.fit()

    Notes:
    ------
    - The class is designed to work seamlessly with statsmodels models.
    - If using "both" as the direction, the "f-test" criterion is not available.
    - Ensure that the data provided is appropriate for the model chosen.

    """

    def __init__(
        self,
        formula: str,
        data: DataFrame,
        model: Union[GLM, OLS, Logit, LogisticRegression] = OLS,
        direction: Literal["both", "forward", "backward"] = "forward",
        criterion: Literal["aic", "bic", "f-test"] = "aic",
        *,
        alpha: float = 0.05,
        max_iter: int = 100,
        formula_params: Dict[str, Any] = dict(),
        fit_params: Dict[str, Any] = {"disp": 0},
        verbose: bool = True,
    ):
        if criterion not in ["aic", "bic", "f-test"]:
            raise ValueError("criterion must be one of 'aic', 'bic', or 'f-test'")
        if not isinstance(formula, str):
            raise TypeError("formula must be a string")
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if direction not in ["both", "forward", "backward"]:
            raise ValueError(
                "direction must be one of 'both', 'forward', or 'backward'"
            )
        if not isinstance(formula_params, dict):
            raise TypeError("formula_params must be a dictionary")
        if not isinstance(fit_params, dict):
            raise TypeError("fit_params must be a dictionary")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        if direction == "both" and criterion == "f-test":
            raise ValueError(
                "direction 'both' is not available with criterion 'f-test'"
            )
        if not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer")
        if max_iter < 1:
            raise ValueError("max_iter must be greater than 0")

        self.formula = formula
        self.data = data
        self.model = model
        self.direction = direction
        self.criterion = criterion
        self.alpha = alpha
        self.max_iter = max_iter
        self.formula_params = formula_params
        self.fit_params = fit_params
        self.verbose = verbose

        self.__vr = self.formula.split("~")[0].strip()
        self.__data_vars = self.data.drop(columns=self.__vr).columns.tolist()
        if self.direction in ["backward", "both"]:
            preds = self.formula.split("~")[1].split("+")
            preds = [x for x in preds if x not in ["-1", "0"]]
        else:
            preds = self.__data_vars
        self.preds = tuple([x.strip() for x in preds])

        self.__metric_name = (
            "p-value" if self.criterion == "f-test" else self.criterion.upper()
        )

    def __forward_selection(self):
        m_actual = self.model.from_formula(
            self.formula, self.data, **self.formula_params
        ).fit(**self.fit_params)
        preds = self.formula.split("~")[1].split("+")
        preds = [x.strip() for x in preds]
        preds = [x for x in preds if x not in ["-1", "0"]]

        self.__BEST_METRICS = []

        termino = False
        cant = sum([1 for x in preds if x in self.data.columns])
        if cant == self.data.shape[1] - 1:
            termino = True
            MIN_METRIC = "Not calculated"

        if self.verbose and self.criterion in ["aic", "bic"]:
            print(
                f"Starting {self.__metric_name}: {m_actual.__getattribute__(self.criterion):0.4f}"
            )

        self.alpha = (
            m_actual.__getattribute__(self.criterion)
            if self.criterion in ["aic", "bic"]
            else self.alpha
        )
        remaining_preds = [x for x in self.preds if x not in self.formula]
        cant_preds = len(remaining_preds)

        f_actual = self.formula

        while not termino:
            METRICS = []
            for p in remaining_preds:
                f_prueba = f"{f_actual} + {p}"
                m_prueba = self.model.from_formula(
                    f_prueba, self.data, **self.formula_params
                ).fit(**self.fit_params)
                if self.criterion in ["aic", "bic"]:
                    METRIC = m_prueba.__getattribute__(self.criterion)
                else:
                    METRIC = nested_models_test(m_actual, m_prueba).p_value
                METRICS.append(METRIC)
            MIN_METRIC = min(METRICS)
            if MIN_METRIC >= self.alpha:
                self.__BEST_METRICS.append(MIN_METRIC)
                self.optimal_metric_ = MIN_METRIC
                MIN_METRIC = self.alpha
                termino = True
                m_actual = self.model.from_formula(
                    f_actual, self.data, **self.formula_params
                ).fit(**self.fit_params)
            else:
                self.alpha = (
                    MIN_METRIC if self.criterion in ["aic", "bic"] else self.alpha
                )
                var_min = remaining_preds[METRICS.index(MIN_METRIC)]
                f_actual = f"{f_actual} + {var_min}"
                options = [" 1 +", " + 1", "+ 1", "+1"]
                for option in options:
                    if option in f_actual:
                        f_actual = f_actual.replace(option, "")
                m_actual = self.model.from_formula(
                    f_actual, self.data, **self.formula_params
                ).fit(**self.fit_params)
                remaining_preds.remove(var_min)
                termino = not bool(remaining_preds)
                if self.verbose:
                    print(
                        f'- Term added: "{var_min}" | {self.__metric_name}: {MIN_METRIC:0.4f}'
                    )

        if self.verbose:
            mm_display = (
                f"{MIN_METRIC:0.4f}" if isinstance(MIN_METRIC, float) else MIN_METRIC
            )
            print(
                bcolors.OKGREEN
                + bcolors.UNDERLINE
                + bcolors.BOLD
                + "Forward selection completed"
                + bcolors.ENDC
            )
            if self.criterion in ["aic", "bic"]:
                print(f"- Obtained {self.__metric_name}: {mm_display}")
            print(
                "- Added terms:",
                cant_preds - len(remaining_preds) if remaining_preds else "None",
            )
            try:
                print(f'- Obtained formula: "{m_actual.model.formula}"')
            except:
                print(f'- Obtained formula: "{m_actual.formula}"')

        return m_actual

    def __backward_selection(self):
        m_actual = self.model.from_formula(
            self.formula, self.data, **self.formula_params
        ).fit(**self.fit_params)
        preds = self.formula.split("~")[1].split("+")
        preds = [x.strip() for x in preds]
        preds = [x for x in preds if x not in ["-1", "0", "1"]]
        self.__BEST_METRICS = []
        termino = False
        cant = len(preds)
        if cant == 0:
            termino = True
            MIN_METRIC = "Not calculated"

        if self.verbose and self.criterion in ["aic", "bic"]:
            print(
                f"Starting {self.__metric_name}: {m_actual.__getattribute__(self.criterion):0.4f}"
            )

        self.alpha = (
            m_actual.__getattribute__(self.criterion)
            if self.criterion in ["aic", "bic"]
            else -self.alpha
        )
        remaining_preds = [x for x in preds]
        cant_preds = len(remaining_preds)

        f_actual = self.formula

        only_one = False
        while not termino:
            METRICS = []
            for p in remaining_preds:
                p_prueba = [x for x in remaining_preds if x != p]
                f_prueba = f"{self.__vr} ~ {' + '.join(p_prueba) if len(p_prueba) != 0 else '1'}"
                if not only_one:
                    m_prueba = self.model.from_formula(
                        f_prueba, self.data, **self.formula_params
                    ).fit(**self.fit_params)
                else:
                    m_prueba = m_prueba
                if len(p_prueba) == 0:
                    only_one = True
                if self.criterion in ["aic", "bic"]:
                    METRIC = m_prueba.__getattribute__(self.criterion)
                else:
                    METRIC = -nested_models_test(m_prueba, m_actual).p_value
                METRICS.append(METRIC)
            MIN_METRIC = min(METRICS)
            if MIN_METRIC >= self.alpha:
                MIN_METRIC = self.alpha
                termino = True
                m_actual = self.model.from_formula(
                    f_actual, self.data, **self.formula_params
                ).fit(**self.fit_params)
            else:
                mm = MIN_METRIC if self.criterion != "f-test" else -MIN_METRIC
                self.__BEST_METRICS.append(mm)
                self.optimal_metric_ = mm
                self.alpha = (
                    MIN_METRIC if self.criterion in ["aic", "bic"] else self.alpha
                )
                var_min = remaining_preds[METRICS.index(MIN_METRIC)]
                remaining_preds.remove(var_min)
                f_actual = f"{self.__vr} ~ {' + '.join(remaining_preds) if remaining_preds else '1'}"
                options = [" 1 +", " + 1", "+ 1", "+1"]
                for option in options:
                    if option in f_actual:
                        f_actual = f_actual.replace(option, "")
                m_actual = self.model.from_formula(
                    f_actual, self.data, **self.formula_params
                ).fit(**self.fit_params)
                termino = not bool(remaining_preds)
                if self.verbose:
                    mm_display = (
                        -MIN_METRIC if self.criterion == "f-test" else MIN_METRIC
                    )
                    print(
                        f'- Term dropped: "{var_min}" | {self.__metric_name}: {mm_display:0.4f}'
                    )

        if self.verbose:
            mm_display = (
                f"{MIN_METRIC:0.4f}" if isinstance(MIN_METRIC, float) else MIN_METRIC
            )
            print(
                bcolors.OKGREEN
                + bcolors.UNDERLINE
                + bcolors.BOLD
                + "Backward selection completed"
                + bcolors.ENDC
            )
            if self.criterion in ["aic", "bic"]:
                print(f"- Obtained {self.__metric_name}: {mm_display}")
            print("- Dropped terms:", cant_preds - len(remaining_preds))
            try:
                print(f'- Obtained formula: "{m_actual.model.formula}"')
            except:
                print(f'- Obtained formula: "{m_actual.formula}"')

        return m_actual

    def __both_selection(self):
        fi = self.formula
        e = self.model.from_formula(fi, self.data, **self.formula_params)
        m = e.fit(**self.fit_params)
        preds = e.formula.split("+")
        data_vars = self.data.drop(columns=self.__vr).columns.tolist()
        preds[0] = preds[0].split("~")[1]
        preds = [x.strip() for x in preds]
        for x in ["0", "-1", "1"]:
            if x in preds:
                preds.remove(x)
        d = self.data.copy()
        data_preds = d.drop(columns=[self.__vr]).columns.tolist()
        METRICF = m.__getattribute__(self.criterion)
        self.__BEST_METRICS = [METRICF]
        METRIC0 = 0

        var_el = []
        var_ag = []

        if self.verbose:
            print(f"Starting {self.criterion.upper()}: {METRICF:0.4f}")

        for i in range(self.max_iter):
            if i == 0:
                METRICS = []
                if len(data_preds) != len(preds):
                    preds0 = preds.copy()
                    data0 = d.copy()
                    ff = f"{self.__vr} ~ {' + '.join(preds0) if preds0 else 1}"
                    continue
                for p in preds:
                    p0 = d.drop(columns=[p, self.__vr]).columns.tolist()
                    f0 = f"{self.__vr} ~ {' + '.join(p0) if p0 else 1}"
                    m0 = self.model.from_formula(f0, d, **self.formula_params).fit(
                        **self.fit_params
                    )
                    METRICS.append(m0.__getattribute__(self.criterion))
                METRIC0 = min(METRICS)
                id_min = METRICS.index(min(METRICS))
                if METRIC0 < METRICF:
                    self.__BEST_METRICS.append(METRIC0)
                    self.optimal_metric_ = METRIC0
                    action = "dropped"
                    var_min = preds[id_min]
                    data0 = d.drop(columns=var_min).copy()
                    preds0 = data0.drop(columns=self.__vr).columns.tolist()
                    ff = f"{self.__vr} ~ {' + '.join(preds0) if preds0 else 1}"
                    var_el.append(var_min)
                    m = self.model.from_formula(ff, d, **self.formula_params).fit(
                        **self.fit_params
                    )
                else:
                    if self.verbose:
                        print(
                            f"There are no improvements in {self.criterion.upper()}, so no terms will be dropped."
                        )
                    return m
            else:
                METRIC0 = METRICF
                METRICS_DROP = []
                for p in preds0:
                    p0 = data0.drop(columns=[p, self.__vr]).columns.tolist()
                    f0 = f"{self.__vr} ~ {' + '.join(p0) if p0 else 1}"
                    m0 = self.model.from_formula(f0, d, **self.formula_params).fit(
                        **self.fit_params
                    )
                    METRICS_DROP.append(m0.__getattribute__(self.criterion))
                METRICS_ADD = []
                for p in data_vars:
                    f0 = f"{ff} {('+ ' + p) if p not in preds0 else ''}"
                    m0 = (
                        self.model.from_formula(f0, d, **self.formula_params).fit(
                            **self.fit_params
                        )
                        if p not in preds0
                        else None
                    )
                    METRICS_ADD.append(
                        m0.__getattribute__(self.criterion)
                        if p not in preds0
                        else m.__getattribute__(self.criterion)
                    )
                MIN_ADD = min(METRICS_ADD)
                MIN_DROP = min(METRICS_DROP) if METRICS_DROP else MIN_ADD
                METRIC0 = MIN_ADD if MIN_ADD < MIN_DROP else MIN_DROP
                if METRIC0 < METRICF:
                    self.__BEST_METRICS.append(METRIC0)
                    self.optimal_metric_ = METRIC0
                    if MIN_DROP < MIN_ADD:
                        action = "dropped"
                        var_min = preds0[METRICS_DROP.index(MIN_DROP)]
                        data0.drop(columns=var_min, inplace=True)
                        preds0.remove(var_min)
                        ff = f"{self.__vr} ~ {' + '.join(preds0) if preds0 else 1}"
                        var_el.append(var_min)
                    else:
                        action = "added"
                        var_min = data_vars[METRICS_ADD.index(MIN_ADD)]
                        data0 = d[preds0 + [var_min, self.__vr]].copy()
                        preds0.append(var_min)
                        ff = f"{self.__vr} ~ {' + '.join(preds0) if preds0 else 1}"
                        var_ag.append(var_min)
                    m = self.model.from_formula(ff, d, **self.formula_params).fit(
                        **self.fit_params
                    )
                    METRICF = METRIC0
                else:
                    if self.verbose:
                        print(
                            bcolors.OKGREEN
                            + bcolors.UNDERLINE
                            + bcolors.BOLD
                            + "Both selection completed"
                            + bcolors.ENDC
                        )
                        print(f"- Obtained {self.criterion.upper()}: {METRICF:0.4f}")
                        print("- Dropped terms:", len(var_el))
                        print("- Added terms:", len(var_ag))
                        print(f'- Obtained formula: "{ff}"')
                    return m
            if self.verbose:
                print(
                    f'- Term "{var_min}" {action} | {self.criterion.upper()}: {METRIC0:0.4f}'
                )
            if not preds0:
                print(
                    bcolors.OKGREEN
                    + bcolors.UNDERLINE
                    + bcolors.BOLD
                    + "Both selection completed"
                    + bcolors.ENDC
                )
                print(f"- Obtained {self.criterion.upper()}: {METRICF:0.4f}")
                print("- Dropped terms:", len(var_el))
                print("- Added terms:", len(var_ag))
                print(f'- Obtained formula: "{ff}"')
                return m
        if self.verbose:
            print("Maximun number of iterations reached")
        return m

    def fit(self):
        """
        Conducts the stepwise process based on the specified direction and criterion.
        Once the stepwise process is completed, the optimal model, formula, and variables
        are stored as instance attributes.

        Examples:
        ---------
        >>> import pandas as pd
        >>> from statsmodels.api import OLS
        >>> from estyp.linear_model import Stepwise
        >>> data = pd.DataFrame({"y": [1,2,3,4,5], "x1": [5,20,3,2,1], "x2": [6,7,8,9,10]})
        >>> stepwise = Stepwise(formula="y ~ 1", data=data, model=OLS, direction="forward", criterion="aic")
        >>> stepwise.fit()
        >>> stepwise.optimal_variables_
        """
        if self.direction == "both":
            final_model = self.__both_selection()
        elif self.direction == "forward":
            final_model = self.__forward_selection()
        else:
            final_model = self.__backward_selection()

        self.optimal_model_ = final_model
        try:
            self.optimal_formula_ = final_model.model.formula
            self.optimal_variables_ = final_model.model.exog_names
        except:
            self.optimal_formula_ = final_model.formula
            self.optimal_variables_ = final_model.params.index.tolist()
        if "Intercept" in self.optimal_variables_:
            self.optimal_variables_.remove("Intercept")

    def plot_history(self, ax=None):
        """
        Plots the history of the chosen criterion (e.g., AIC, BIC, or F-test) during the stepwise process.
        This provides a visual representation of how the criterion changes as predictors are added or removed
        from the model.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional (default = None)
            An Axes instance for the plot. If not provided, a new figure and axes will be created.

        Returns:
        --------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            The Figure and Axes instances containing the plot if not provided.

        Examples:
        ---------
        >>> import pandas as pd
        >>> from statsmodels.api import OLS
        >>> from estyp.linear_model import Stepwise
        >>> data = pd.DataFrame({"y": [1,2,3,4,5], "x1": [5,20,3,2,1], "x2": [6,7,8,9,10]})
        >>> stepwise = Stepwise(formula="y ~ x1 + x2", data=data, model=OLS, direction="forward", criterion="aic")
        >>> stepwise.fit()
        >>> fig, ax = stepwise.plot_history()"""
        if ax is not None and not isinstance(ax, plt.Axes):
            raise TypeError("ax must be an instance of matplotlib.axes.Axes")

        l = len(self.__BEST_METRICS)
        x = range(1, l + 1)

        if ax is None:
            fig, ax = plt.subplots()
            ax.plot(x, self.__BEST_METRICS, marker="o")
            ax.set_xticks(x)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(self.criterion.upper())
            ax.set_title(
                f'Stepwise with direction="{self.direction}" and criterion="{self.criterion}"'
            )
            return fig, ax

        else:
            ax.plot(x, self.__BEST_METRICS, marker="o")
            ax.set_xticks(x)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(self.criterion.upper())
            ax.set_title(
                f'Stepwise with direction="{self.direction}" and criterion="{self.criterion}"'
            )

    def __repr__(self) -> str:
        return f'Stepwise"(direction={self.direction}", criterion="{self.criterion}")'
