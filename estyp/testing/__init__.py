from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from statsmodels.regression.linear_model import RegressionResultsWrapper

from estyp.testing.__base import (TestResults, _CheckHomocedasticity,
                                    _CheckIndependence,
                                    _CheckMulticollinearity, _CheckNormality,
                                    __nested_models_test, __prop_test, __t_test,
                                    __var_test)

class CheckModel:
    """
    # Check Linear Regression Assumptions

    The `CheckModel` class provides methods to test the assumptions of the linear regression model.
    These assumptions are:

    - Normality of residuals
    - Homoscedasticity (equal variance) of residuals
    - Independence of residuals
    - No Multicollinearity among predictors

    Parameters
    ----------
    `fitted_model` : `RegressionResultsWrapper`
        The fitted linear regression model which is an instance of RegressionResultsWrapper.

    Example
    -------
    ```python
    import statsmodels.api as sm
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X = diabetes["data"]
    y = diabetes["target"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    fitted_model = model.fit()
    cm = CheckModel(fitted_model)
    cm.check_all()
    ```
    """

    def __init__(self, fitted_model: RegressionResultsWrapper):
        if not isinstance(fitted_model, RegressionResultsWrapper):
            raise TypeError(
                "The fitted model must be a RegressionResultsWrapper object"
            )
        self.model = fitted_model

    def check_normality(self, alpha=0.05, plot=True, return_pvals=False):
        """
        Checks the normality assumption of the residuals using several statistical tests.

        Parameters
        ----------
        `alpha` : float, optional
            The significance level used by the statistical tests, by default 0.05.
        `plot` : bool, optional
            If `True`, a plot is shown for visual inspection of normality, by default `True`.
        `return_pvals` : bool, optional
            If `True`, the p-values of the statistical tests are returned, by default `False`.

        Returns
        -------
        dict
            A dictionary of p-values of the statistical tests if `return_pvals=True`.

        Example
        -------
        >>> cm.check_normality()
        """
        normality = _CheckNormality(self.model)
        if plot:
            fig, ax = normality.plot()
            plt.show()
        pvals = normality.test(sign_level=alpha)
        if return_pvals:
            return pvals

    def check_homocedasticity(self, alpha=0.05, plot=True, return_pvals=False):
        """
        Checks the homoscedasticity assumption (equal variance of residuals) using several statistical tests.
        
        Parameters
        ----------
        `alpha` : float, optional
            The significance level used by the statistical tests, by default 0.05.
        `plot` : bool, optional
            If `True`, a plot is shown for visual inspection of homoscedasticity, by default `True`.
        `return_pvals` : bool, optional
            If `True`, the p-values of the statistical tests are returned, by default `False`.
        
        Returns
        -------
        dict
            A dictionary of p-values of the statistical tests if `return_pvals=True`.
            
        Example
        -------
        >>> cm.check_homocedasticity()
        """
        homocedasticity = _CheckHomocedasticity(self.model)
        if plot:
            fig, ax = homocedasticity.plot()
            plt.show()
        pvals = homocedasticity.test(sign_level=alpha)
        if return_pvals:
            return pvals

    def check_independence(self, alpha=0.05, plot=True, return_vals=False):
        """
        Checks the independence assumption of the residuals using several statistical tests.
        
        Parameters
        ----------
        alpha : float, optional
            The significance level used by the statistical tests, by default 0.05.
        plot : bool, optional
            If `True`, a plot is shown for visual inspection of independence, by default `True`.
        return_pvals : bool, optional
            If `True`, the p-values of the statistical tests are returned, by default `False`.
        
        Returns
        -------
        dict
            A dictionary of values of the statistical tests if `return_vals=True`.
            
        Example
        -------
        >>> cm.check_independence()
        """
        independence = _CheckIndependence(self.model)
        if plot:
            fig, ax = independence.plot()
            plt.show()
        vals = independence.test(sign_level=alpha)
        if return_vals:
            return vals

    def check_multicollinearity(self, plot=True, return_cm=False):
        """
        Checks the multicollinearity assumption among predictors using the variance inflation factor (VIF).
        
        Parameters
        ----------
        `plot` : bool, optional
            If True, a plot is shown for visual inspection of multicollinearity, by default True.
        `return_cm` : bool, optional
            Returns the condition number of the model if True, by default False.
            
        Returns
        -------
        float
            The condition number of the model if `return_cm=True`.
        
        Example
        -------
        >>> cm.check_multicollinearity()
        """
        multicollinearity = _CheckMulticollinearity(self.model)
        if plot:
            fig, ax = multicollinearity.plot()
            plt.show()
        pvals = multicollinearity.test()
        if return_cm:
            return pvals

    def check_all(self, alpha=0.05, plot=True, return_vals=False):
        """
        Checks all the assumptions of the linear regression model.
        
        Parameters
        ----------
        `alpha` : float, optional
            The significance level used by the statistical tests, by default 0.05.
        `plot` : bool, optional
            If `True`, a plot is shown for visual inspection of each assumption, by default `True`.
        `return_vals` : bool, optional
            If `True`, the values of the statistical tests are returned, by default `False`.
        
        Returns
        -------
        dict
            A dictionary of values of the statistical tests if `return_vals=True`.
            
        Example
        -------
        >>> cm.check_all()
        """
        normality = _CheckNormality(self.model)
        homocedasticity = _CheckHomocedasticity(self.model)
        independence = _CheckIndependence(self.model)
        multicollinearity = _CheckMulticollinearity(self.model)

        normality_results = normality.test(sign_level=alpha)
        if plot:
            fig, ax = normality.plot()
            plt.show()
        homocedasticity_results = homocedasticity.test(sign_level=alpha)
        if plot:
            fig, ax = homocedasticity.plot()
            plt.show()
        independence_results = independence.test(sign_level=alpha)
        if plot:
            fig, ax = independence.plot()
            plt.show()
        multicollinearity_results = multicollinearity.test()
        if plot:
            fig, ax = multicollinearity.plot()
            plt.show()
        names = ["Normality", "Homocedasticity", "Independence", "Multicollinearity"]
        tests = [
            normality_results,
            homocedasticity_results,
            independence_results,
            multicollinearity_results,
        ]
        if return_vals:
            return dict(zip(names, tests))


def var_test(
    x: Union[List, np.ndarray, Series],
    y: Union[List, np.ndarray, Series],
    ratio: Union[float, int]=1,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    conf_level=0.95,
) -> TestResults:
    """
    # F Test to Compare Two Variances
    ## Description
    Performs an F test to compare the variances of two samples from normal populations. This function is inspired by the `var.test()` function of the software R.

    ## Usage

    ```python
    var_test(x, y, ...)
    var_test(x, y, ratio = 1,
            alternative = [two-sided", "less", "greater"],
            conf_level = 0.95)
    ```

    ## Arguments
    * `x`, `y`: numeric list, `np.array` or `pd.Series` of data values.
    * `ratio`: the hypothesized ratio of the population variances of x and y.
    * `alternative`: a character string specifying the alternative hypothesis, must be one of "two.sided" (default), "greater" or "less". You can specify just the initial letter.
    * `conf_level`: a number between 0 and 1 indicanting the confidence level of the interval.


    ## Details
    The null hypothesis is that the ratio of the variances of the populations from which x and y were drawn, is equal to ratio.

    ## Value
    An instance of the `TestResults` class containing the following attributes:

    * `statistic`: the value of the F test statistic.
    * `df`: the degrees of freedom for the F test statistic.
    * `p_value`: the p-value for the test.
    * `ci`: a confidence interval for the ratio of the population variances.
    * `estimate`: the ratio of the sample variances of x and y.
    * `alternative`: a string describing the alternative hypothesis.

    ## Examples

    ```python
    import numpy as np
    np.random.seed(2023)
    x = np.random.normal(size=100)
    y = np.random.normal(size=100)

    print(var_test(x, y))
    print(var_test(x, y, alternative="less"))
    print(var_test(x, y, ratio = 0.9, alternative="greater"))
    ```
    """
    return __var_test(x, y, ratio, alternative, conf_level)


def t_test(
    x: Union[List, np.ndarray, Series],
    y: Union[List, np.ndarray, Series] = None,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    mu: Union[float, int] = 0,
    paired=False,
    var_equal=False,
    conf_level=0.95,
) -> TestResults:
    """

    # Student's t-Test
    ## Description
    Performs one and two sample t-tests on groups of data. This function is inspired by the `t.test()` function of the software R.

    ## Usage
    ```python
    t_test(x, ...)
    ```

    ```python
    t_test(x, y = None,
        alternative = ["two-sided", "less", "greater"],
        mu = 0, paired = False, var_equal = False,
        conf_level = 0.95)
    ```


    ## Arguments
    * `x`: a (non-empty) numeric container of data values.
    * `y`: an (optional) numeric container of data values.
    * `alternative`: a string specifying the alternative hypothesis, must be one of "two-sided" (default), "greater" or "less".
    * `mu`: a number indicating the true value of the mean (or difference in means if you are performing a two sample test).
    * `paired`: a logical indicating whether you want a paired t-test.
    * `var_equal`: a logical variable indicating whether to treat the two variances as being equal. If `True` then the pooled variance is used to estimate the variance otherwise the Welch (or Satterthwaite) approximation to the degrees of freedom is used.
    * `conf_level`: a number between 0 and 1 indicanting the confidence level of the interval.


    ## Details
    alternative = "greater" is the alternative that x has a larger mean than y. For the one-sample case: that the mean is positive.

    If `paired` is `True` then both x and y must be specified and they must be the same length. Missing values are silently removed (in pairs if `paired` is `True`). If `var_equal` is `True` then the pooled estimate of the variance is used. By default, if `var_equal` is `False` then the variance is estimated separately for both groups and the Welch modification to the degrees of freedom is used.


    ### Value
    An instance of the `TestResults` class containing the following attributes:

    * `statistic`: the value of the t-statistic.
    * `df`: the degrees of freedom for the t-statistic.
    * `p_value`: the p-value for the test.
    * `ci`: a confidence interval for the mean appropriate to the specified alternative hypothesis.
    * `estimate`: the estimated mean or list of estimated means depending on whether it was a one-sample test or a two-sample test.
    * `alternative`: a character string describing the alternative hypothesis.
    * `mu`: the mean of the null hypothesis.


    ## Examples
    ```python
    import numpy as np
    np.random.seed(2023)
    x = np.random.normal(size=100)
    y = np.random.normal(size=100)
    mu = 0.1

    print(t_test(x, mu=mu, alternative="less"))
    print(t_test(x, y, mu=mu))
    print(t_test(x, y, mu=mu, var_equal=True, alternative="greater"))
    print(t_test(x, y, mu=mu, paired=True))
    ```
    """
    return __t_test(x, y, alternative, mu, paired, var_equal, conf_level)


def nested_models_test(
    fitted_small_model: RegressionResultsWrapper,
    fitted_big_model: RegressionResultsWrapper,
) -> TestResults:
    """
    # Nested Models F-Test Function

    ## Description

    This function performs a nested models F-test using deviance from two fitted models from statsmodels library. The test compares two nested models: a larger or "big" model and a smaller or "small" model. The purpose of this test is to determine whether the larger model significantly improves the model fit compared to the smaller model by adding additional predictors.

    ## Parameters
    * `fitted_small_model`: The fitted model representing the smaller/nested model. It have to come from statsmodels.
    * `fitted_big_model`: : The fitted model representing the larger model, which includes all the predictors from the smaller model and potentially additional predictors. It have to come from statsmodels.

    ## Returns

    The function returns an object of class TestResults that contains the following information:

    * `method`: A string indicating the name of the statistical test (Nested models F-test).
    * `statistic`: The computed F-statistic value.
    * `estimate`: The difference in deviances between the models.
    * `df`: A dictionary with the degrees of freedom for the numerator and denominator of the F-statistic.
    * p_value: The p-value associated with the F-statistic.

    ## Examples
    ### Example 1: With OLS
    ```python
    import pandas as pd
    import statsmodels.api as sm
    data = pd.DataFrame({
        "x": [2.01, 2.99, 4.01, 5.01, 6.89],
        "y": [2, 3, 4, 5, 6]
    })

    model_small = sm.OLS.from_formula("y ~ 1", data).fit()
    model_big = sm.OLS.from_formula("y ~ x", data).fit()

    print(nested_models_test(model_small, model_big))
    ```
    ### Example 2: With Logit
    ```python
    import pandas as pd
    import statsmodels.api as sm
    data = pd.DataFrame({
        "x": [2.01, 2.99, 4.01, 3.01, 4.89],
        "y": [0, 1, 1, 0, 1]
    })

    model_small = sm.Logit.from_formula("y ~ 1", data).fit()
    model_big = sm.Logit.from_formula("y ~ x", data).fit()

    print(nested_models_test(model_small, model_big))
    ```
    ### Example 3: With GLM
    ```python
    import pandas as pd
    import statsmodels.api as sm
    data = pd.DataFrame({
        "x": [2.01, 2.99, 4.01, 5.01, 6.89],
        "y": [2, 3, 4, 5, 6]
    })
    model_small = sm.GLM.from_formula("y ~ 1", data, family = sm.families.Gamma()).fit()
    model_big = sm.GLM.from_formula("y ~ x", data, family = sm.families.Gamma()).fit()

    print(nested_models_test(model_small, model_big))
    ```
    """
    return __nested_models_test(fitted_small_model, fitted_big_model)


def prop_test(
    x: Union[List, np.ndarray, Series],
    n: Optional[Union[List, np.ndarray, Series]] = None,
    p: Optional[Union[float, List, np.ndarray]] = None,
    alternative: str = "two-sided",
    conf_level: float = 0.95,
    correct: bool = True,
) -> TestResults:
    """
    # Test of Equal or Given Proportions.

    `prop_test()` can be used for testing the null that the proportions
    (probabilities of success) in several groups are the same, or that they equal
    certain given values.

    Parameters
    ----------
    `x` : array_like
        a vector of counts of successes, a one-dimensional table with two entries,
        or a two-dimensional table (or matrix) with 2 columns, giving the counts of
        successes and failures, respectively.
    `n` : array_like, optional
        a vector of counts of trials; ignored if x is a matrix or a table.
        If not provided, it is calculated as the sum of the elements in x.
    `p` : array_like, optional
        a vector of probabilities of success. The length of p must be the same as
        the number of groups specified by x, and its elements must be greater than
        0 and less than 1.
    alternative : str, optional
        a character string specifying the alternative hypothesis, must be one of
        "two.sided" (default), "greater" or "less". You can specify just the
        initial letter. Only used for testing the null that a single proportion
        equals a given value, or that two proportions are equal; ignored otherwise.
    `conf_level` : float, optional
        confidence level of the returned confidence interval. Must be a single
        number between 0 and 1. Only used when testing the null that a single
        proportion equals a given value, or that two proportions are equal;
        ignored otherwise.
    `correct` : bool, optional
        a logical indicating whether Yates' continuity correction should be
        applied where possible.

    Returns
    -------
    `TestResult`
        A data class with the following attributes:

        `statistic` : float
            the value of Pearson's chi-squared test statistic.
        `df` : int
            the degrees of freedom of the approximate chi-squared distribution of
            the test statistic.
        `p_value` : float
            the p-value of the test.
        `estimate` : array_like
            a vector with the sample proportions x/n.
        `null_value` : float or array_like
            the value of p if specified by the null hypothesis.
        `conf_int` : array_like
            a confidence interval for the true proportion if there is one group,
            or for the difference in proportions if there are 2 groups and p is
            not given, or None otherwise. In the cases where it is not None, the
            returned confidence interval has an asymptotic confidence level as
            specified by conf_level, and is appropriate to the specified
            alternative hypothesis.
        `alternative` : str
            a character string describing the alternative.
        `method` : str
            a character string indicating the method used, and whether Yates'
            continuity correction was applied.

    Example
    --------
    ```python
    import numpy as np
    from scipy import stats
    x = np.array([83, 90, 129, 70])
    n = np.array([86, 93, 136, 82])
    result = prop_test(x, n)
    print(result)
    ```
    """
    return __prop_test(x, n, p, alternative, conf_level, correct)

