from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.api import OLS

from estyp.testing.__base import (
    TestResults,
    _CheckHomocedasticity,
    _CheckIndependence,
    _CheckMulticollinearity,
    _CheckNormality,
    __nested_models_test,
    __prop_test,
    __t_test,
    __var_test,
    __cor_test,
    __chisq_test,
    _dw_test,
)


class CheckModel:
    """
Check Linear Regression Assumptions
===================================

View this in the [online documentation](https://estyp.readthedocs.io/en/latest/testing.html#CheckModel).

Description
-----------

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

>>> import statsmodels.api as sm
>>> from sklearn.datasets import load_diabetes
>>> diabetes = load_diabetes()
>>> X = diabetes["data"]
>>> y = diabetes["target"]
>>> X = sm.add_constant(X)
>>> model = sm.OLS(y, X)
>>> fitted_model = model.fit()
>>> cm = CheckModel(fitted_model)
>>> cm.check_all()
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
    *,
    ratio: Union[float, int] = 1,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    conf_level=0.95,
) -> TestResults:
    """
F Test to Compare Two Variances
===============================

View this in the [online documentation](https://estyp.readthedocs.io/en/latest/testing.html#f-test-to-compare-two-variances).

Description
-----------
Performs an F test to compare the variances of two samples from normal populations. This function is inspired by the `var.test()` function of the software R.


Arguments
---------
- `x`, `y`: numeric list, `np.array` or `pd.Series` of data values.
- `ratio`: the hypothesized ratio of the population variances of x and y.
- `alternative`: a character string specifying the alternative hypothesis, must be one of "two-sided" (default), "greater" or "less". You can specify just the initial letter.
- `conf_level`: a number between 0 and 1 indicanting the confidence level of the interval.

Details
-------
The null hypothesis is that the ratio of the variances of the populations from which x and y were drawn, is equal to ratio.

Value
-----
An instance of the `TestResults` class containing the following attributes:

- `statistic`: the value of the F test statistic.
- `df`: the degrees of freedom for the F test statistic.
- `p_value`: the p-value for the test.
- `ci`: a confidence interval for the ratio of the population variances.
- `estimate`: the ratio of the sample variances of x and y.
- `alternative`: a string describing the alternative hypothesis.

Examples
--------

>>> import numpy as np
>>> np.random.seed(2023)
>>> x = np.random.normal(size=100)
>>> y = np.random.normal(size=100)

>>> print(var_test(x, y))
>>> print(var_test(x, y, alternative="less"))
>>> print(var_test(x, y, ratio=0.9, alternative="greater"))
    """
    return __var_test(x, y, ratio, alternative, conf_level)


def t_test(
    x: Union[List, np.ndarray, Series],
    y: Union[List, np.ndarray, Series] = None,
    *,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    mu: Union[float, int] = 0,
    paired=False,
    var_equal=False,
    conf_level=0.95,
) -> TestResults:
    """
Student's t-Test
================

View this in the [online documentation](https://estyp.readthedocs.io/en/latest/testing.html#student-s-t-test).

Description
-----------
Performs one and two sample t-tests on groups of data. This function is inspired by the `t.test()` function of the software R.


Arguments
---------
- `x`: a (non-empty) numeric container of data values.
- `y`: an (optional) numeric container of data values.
- `alternative`: a string specifying the alternative hypothesis, must be one of "two-sided" (default), "greater" or "less".
- `mu`: a number indicating the true value of the mean (or difference in means if you are performing a two sample test).
- `paired`: a logical indicating whether you want a paired t-test.
- `var_equal`: a logical variable indicating whether to treat the two variances as being equal. If `True` then the pooled variance is used to estimate the variance otherwise the Welch (or Satterthwaite) approximation to the degrees of freedom is used.
- `conf_level`: a number between 0 and 1 indicanting the confidence level of the interval.

Details
-------
alternative = "greater" is the alternative that x has a larger mean than y. For the one-sample case: that the mean is positive.

If `paired` is `True` then both x and y must be specified and they must be the same length. Missing values are silently removed (in pairs if `paired` is `True`). If `var_equal` is `True` then the pooled estimate of the variance is used. By default, if `var_equal` is `False` then the variance is estimated separately for both groups and the Welch modification to the degrees of freedom is used.

Value
-----
An instance of the `TestResults` class containing the following attributes:

- `statistic`: the value of the t-statistic.
- `df`: the degrees of freedom for the t-statistic.
- `p_value`: the p-value for the test.
- `ci`: a confidence interval for the mean appropriate to the specified alternative hypothesis.
- `estimate`: the estimated mean or list of estimated means depending on whether it was a one-sample test or a two-sample test.
- `alternative`: a character string describing the alternative hypothesis.
- `mu`: the mean of the null hypothesis.

Examples
--------

>>> import numpy as np
>>> np.random.seed(2023)
>>> x = np.random.normal(size=100)
>>> y = np.random.normal(size=100)
>>> mu = 0.1

>>> print(t_test(x, mu=mu, alternative="less"))
>>> print(t_test(x, y, mu=mu))
>>> print(t_test(x, y, mu=mu, var_equal=True, alternative="greater"))
>>> print(t_test(x, y, mu=mu, paired=True))
    """
    return __t_test(x, y, alternative, mu, paired, var_equal, conf_level)


def nested_models_test(
    fitted_small_model: RegressionResultsWrapper,
    fitted_big_model: RegressionResultsWrapper,
) -> TestResults:
    """
Nested Models F-Test Function
=============================

View this in the [online documentation](https://estyp.readthedocs.io/en/latest/testing.html#nested-models-f-test-function).

Description:
-------------
This function performs a nested models F-test using deviance from two fitted models from statsmodels library. The test compares two nested models: a larger or "big" model and a smaller or "small" model. The purpose of this test is to determine whether the larger model significantly improves the model fit compared to the smaller model by adding additional predictors.

Parameters:
-----------
`fitted_small_model` : `RegressionResultsWrapper`
    The fitted model representing the smaller/nested model. It has to come from statsmodels.
`fitted_big_model` : `RegressionResultsWrapper`
    The fitted model representing the larger model, which includes all the predictors from the smaller model and potentially additional predictors. It has to come from statsmodels.

Returns:
--------
The function returns an object of class TestResults that contains the following information:

`method` : str
    A string indicating the name of the statistical test (Nested models F-test).
`statistic` : float
    The computed F-statistic value.
`estimate` : float
    The difference in deviances between the models.
`df` : dict
    A dictionary with the degrees of freedom for the numerator and denominator of the F-statistic.
`p_value` : float
    The p-value associated with the F-statistic.

Examples:
---------
Example 1: With OLS

>>> import pandas as pd
>>> import statsmodels.api as sm
>>> data = pd.DataFrame({
...     "x": [2.01, 2.99, 4.01, 5.01, 6.89],
...     "y": [2, 3, 4, 5, 6]
... })
>>> model_small = sm.OLS.from_formula("y ~ 1", data).fit()
>>> model_big = sm.OLS.from_formula("y ~ x", data).fit()
>>> print(nested_models_test(model_small, model_big))

Example 2: With Logit

>>> import pandas as pd
>>> import statsmodels.api as sm
>>> data = pd.DataFrame({
...     "x": [2.01, 2.99, 4.01, 3.01, 4.89],
...     "y": [0, 1, 1, 0, 1]
... })
>>> model_small = sm.Logit.from_formula("y ~ 1", data).fit()
>>> model_big = sm.Logit.from_formula("y ~ x", data).fit()
>>> print(nested_models_test(model_small, model_big))

Example 3: With GLM

>>> import pandas as pd
>>> import statsmodels.api as sm
>>> data = pd.DataFrame({
...     "x": [2.01, 2.99, 4.01, 5.01, 6.89],
...     "y": [2, 3, 4, 5, 6]
... })
>>> model_small = sm.GLM.from_formula("y ~ 1", data, family = sm.families.Gamma()).fit()
>>> model_big = sm.GLM.from_formula("y ~ x", data, family = sm.families.Gamma()).fit()
>>> print(nested_models_test(model_small, model_big))
    """
    return __nested_models_test(fitted_small_model, fitted_big_model)


def prop_test(
    x: Union[List, np.ndarray, Series],
    *,
    n: Optional[Union[List, np.ndarray, Series]] = None,
    p: Optional[Union[float, List, np.ndarray]] = None,
    alternative: str = "two-sided",
    conf_level: float = 0.95,
    correct: bool = True,
) -> TestResults:
    """
Test of Equal or Given Proportions.
===================================

View this in the [online documentation](https://estyp.readthedocs.io/en/latest/testing.html#test-of-equal-or-given-proportions).

Parameters
----------
`x` : array_like
    A vector of counts of successes, a one-dimensional table with two entries,
    or a two-dimensional table (or matrix) with 2 columns, giving the counts of
    successes and failures, respectively.
`n` : array_like, optional
    A vector of counts of trials; ignored if x is a matrix or a table.
    If not provided, it is calculated as the sum of the elements in x.
`p` : array_like, optional
    A vector of probabilities of success. The length of p must be the same as
    the number of groups specified by x, and its elements must be greater than
    0 and less than 1.
`alternative` : str, optional
    A character string specifying the alternative hypothesis, must be one of
    "two-sided" (default), "greater" or "less". You can specify just the
    initial letter. Only used for testing the null that a single proportion
    equals a given value, or that two proportions are equal; ignored otherwise.
`conf_level` : float, optional
    Confidence level of the returned confidence interval. Must be a single
    number between 0 and 1. Only used when testing the null that a single
    proportion equals a given value, or that two proportions are equal;
    ignored otherwise.
`correct` : bool, optional
    A logical indicating whether Yates' continuity correction should be
    applied where possible.

Returns
-------
`TestResults`
    A data class with the following attributes:

    - `statistic` : float
        The value of Pearson's chi-squared test statistic.
    - `df` : int
        The degrees of freedom of the approximate chi-squared distribution of
        the test statistic.
    - `p_value` : float
        The p-value of the test.
    - `estimate` : array_like
        A vector with the sample proportions x/n.
    - `null_value` : float or array_like
        The value of p if specified by the null hypothesis.
    - `conf_int` : array_like
        A confidence interval for the true proportion if there is one group,
        or for the difference in proportions if there are 2 groups and p is
        not given, or None otherwise. In the cases where it is not None, the
        returned confidence interval has an asymptotic confidence level as
        specified by conf_level, and is appropriate to the specified
        alternative hypothesis.
    - `alternative` : str
        A character string describing the alternative.
    - `method` : str
        A character string indicating the method used, and whether Yates'
        continuity correction was applied.

Examples
--------

>>> import numpy as np
>>> from scipy import stats
>>> x = np.array([83, 90, 129, 70])
>>> n = np.array([86, 93, 136, 82])
>>> result = prop_test(x, n=n)
>>> print(result)
    """
    return __prop_test(x, n, p, alternative, conf_level, correct)


def cor_test(
    x: Union[List, np.ndarray, Series],
    y: Union[List, np.ndarray, Series],
    *,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    conf_level: float = 0.95,
    continuity: bool = False,
) -> TestResults:
    """
Test for Association/Correlation Between Paired Samples
=======================================================

View this in the [online documentation](https://estyp.readthedocs.io/en/latest/testing.html#test-for-association-correlation-between-paired-samples).

Description
-----------
Test for association between paired samples, using one of Pearson's product moment correlation coefficient, Kendall's
tau or Spearman's rho.

Arguments
---------
`x`, `y` : array_like
    Numeric one-dimensional `arrays`, `lists` or `pd.Series` of data values. `x` and `y` must have the same length.

`alternative` : str, optional
    Indicates the alternative hypothesis and must be one of "two-sided", "greater" or "less".
    "greater" corresponds to positive association, "less" to negative association.

`method` : str, optional
    A string indicating which correlation coefficient is to be used for the test.
    One of "pearson", "kendall", or "spearman".

`conf_level` : float, optional
    Confidence level for the returned confidence interval. Currently only used for the
    Pearson product moment correlation coefficient if there are at least 4 complete pairs of observations.

`continuity` : bool, optional
    If `True`, a continuity correction is used for Kendall's tau.

Returns
-------
A `TestResults` instance containing the following atributes:

- `statistic`: the value of the test statistic.
- `df` (if applicable): the degrees of freedom of the test statistic.
- `p_value`: the p-value of the test.
- `estimate`: the estimated measure of association.
- `null_value`: the value of the association measure under the null hypothesis, always 0.
- `alternative`: a string describing the alternative hypothesis.
- `method`: a string indicating how the association was measured.
- `conf_int` (if applicable): a confidence interval for the measure of association.

Details
-------
The three methods each estimate the association between paired samples and compute a test of the value being zero.
They use different measures of association, all in the range [-1, 1] with 0 indicating no association.
These are sometimes referred to as tests of no correlation, but that term is often confined to the default method.

References
----------
[1] D. J. Best & D. E. Roberts (1975). Algorithm AS 89: The Upper Tail Probabilities of Spearman's rho.
    Applied Statistics, 24, 377--379. 10.2307/2347111.

[2] Myles Hollander & Douglas A. Wolfe (1973), Nonparametric Statistical Methods.
    New York: John Wiley & Sons. Pages 185--194 (Kendall and Spearman tests).

Example
-------
Using the iris dataset to test the association between sepal length and petal length using Pearson's correlation:

>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> sepal_length = iris.data[:, 0]
>>> petal_length = iris.data[:, 2]
>>> result = cor_test(sepal_length, petal_length, method="pearson")
>>> print(result)

    """
    return __cor_test(x, y, method, alternative, conf_level, continuity)


def chisq_test(
    x: Union[List, np.ndarray, Series],
    *,
    y: Union[List, np.ndarray, Series, None] = None,
    p: Union[List, np.ndarray, Series, None] = None,
    correct: bool = True,
    rescale_p: bool = False
):
    """
Pearson's Chi-squared Test for Count Data
=========================================

View this in the [online documentation](https://estyp.readthedocs.io/en/latest/testing.html#pearson-s-chi-squared-test-for-count-data).

Description
-----------
`chisq_test()` performs chi-squared contingency table tests and goodness-of-fit tests.

Arguments
---------
`x`: arrat_like
    A numeric list or 2D list (matrix). x and y can also both be lists.
    
`y`: array_like
    A numeric data ; ignored if x is a matrix. If x is a list, y should be a list of the same length. The default is None.

`p`: array_like
    A list of probabilities of the same length as x. An error is raised if any entry of p is negative.

`correct`: 
    A boolean indicating whether to apply continuity correction when computing the test statistic for 2x2 tables: 
    one half is subtracted from all abs(O-E) differences; however, the correction will not be bigger than the differences themselves. The default is True.

`rescale_p`: boolean
    A boolean; if True then p is rescaled (if necessary) to sum to 1. If rescale_p is False, and p does not sum to 1, an error is raised.

Details
-------
    If x is a matrix with one row or column, or if x is a list and y is not given, then a goodness-of-fit test is performed 
    (x is treated as a one-dimensional contingency table). The entries of x must be non-negative integers. In this case, the hypothesis 
    tested is whether the population probabilities equal those in p, or are all equal if p is not given.

    If x is a matrix with at least two rows and columns, it is taken as a two-dimensional contingency table: the entries of x must be 
    non-negative integers. Otherwise, x and y must be lists of the same length; cases with None values are removed, the lists are 
    treated as factors, and the contingency table is computed from these. Then Pearson's chi-squared test is performed of the null 
    hypothesis that the joint distribution of the cell counts in a 2-dimensional contingency table is the product of the row and column marginals.

    The p-value is computed from the asymptotic chi-squared distribution of the test statistic; continuity correction is only used in 
    the 2-by-2 case (if correct is True, the default).

Returns
-------
    A `TestResults` instance containing the following attributes:

    `statistic`: 
        The value of the chi-squared test statistic.

    `df`: 
        The degrees of freedom of the approximate chi-squared distribution of the test statistic.

    `p_value`:
        The p-value for the test.

    `method`:
        A string indicating the type of test performed, and whether continuity correction was used.

    `expected`:
        The expected counts under the null hypothesis.


Examples
--------

From Agresti(2007) p.39

>>> M = [[762, 327, 468], [484, 239, 477]]
>>> result1 = chisq_test(M)
>>> print(result1)

Effect of rescale_p

>>> x = [12, 5, 7, 7]
>>> p = [0.4, 0.4, 0.2, 0.2]
>>> result2 = chisq_test(x, p=p, rescale_p=True)
>>> print(result2)

Testing for population probabilities

>>> x = [20, 15, 25]
>>> result31 = chisq_test(x)
>>> print(result31)

A second example of testing for population probabilities

>>> x = [89,37,30,28,2]
>>> p = [0.40,0.20,0.20,0.19,0.01]
>>> result32 = chisq_test(x, p=p)
>>> print(result32)

Goodness of fit

>>> x = [1, 2, 3, 4, 5, 6]
>>> y = [6, 1, 2, 3, 4, 5]
>>> result4 = chisq_test(x, y=y)
>>> print(result4)
    """

    return __chisq_test(x, y, p, correct, rescale_p)

def dw_test(
        input_obj: Union[str, OLS, RegressionResultsWrapper],
        *,
        order_by:Union[Series, np.ndarray] = None,
        alternative: Literal["two-sided", "greater", "less"] = "greater",
        data: DataFrame = None
) -> TestResults:
    """
Durbin-Watson Test
==================

View this in the [online documentation](https://estyp.readthedocs.io/en/latest/testing.html#durbin-watson-test-for-autocorrelation).

Description
-----------

Performs the Durbin-Watson test for autocorrelation of disturbances. Includes an approximated p-value taken from the `lmtest::dwtest()` function in R.


Parameters:
-----------

`input_obj`: One of str, OLS, or a fitted OLS object. 
        A formula description for the model to be tested (or a OLS object or a fitted OLS object).
`order_by`: Series, np.ndarray, optional
        The observations in the model are ordered by z. 
        If set to None (the default) the observations are assumed to be ordered (e.g., a time series).
`alternative`: str, optional.
        A character string specifying the alternative hypothesis. One of "greater" (default), "two-sided", or "less".
`data` : pd.DataFrame, optional 
        An optional pandas DataFrame containing the variables in the model.

Details
-------

The Durbin-Watson test has the null hypothesis that the autocorrelation of the disturbances is 0. 
It is possible to test against the alternative that it is greater than, not equal to, or less than 0, respectively. 
This can be specified by the alternative parameter.

For large sample sizes the algorithm might fail to compute the p value; in that case a warning is printed 
and an approximate p value will be given; this p value is computed using a normal approximation 
with mean and variance of the Durbin-Watson test statistic.

Returns
-------

A `TestResults` instance containing:

- `statistic`: the test statistic.
- `method`: a character string with the method used.
- `alternative`: a character string describing the alternative hypothesis.
- `p_value`: the corresponding p-value.
` `estimate`: the estimate of the autocorrelation of firt order of the residuals obtained from DW statistic.

References
----------

- J. Durbin & G.S. Watson (1950), Testing for Serial Correlation in Least Squares Regression I. Biometrika 37, 409–428.
- J. Durbin & G.S. Watson (1951), Testing for Serial Correlation in Least Squares Regression II. Biometrika 38, 159–177.
- J. Durbin & G.S. Watson (1971), Testing for Serial Correlation in Least Squares Regression III. Biometrika 58, 1–19.
- R.W. Farebrother (1980), Pan's Procedure for the Tail Probabilities of the Durbin-Watson Statistic. 
    Applied Statistics 29, 224–227.
- W. Krämer & H. Sonnberger (1986), The Linear Regression Model under Test. Heidelberg: Physica.

Examples
--------

Generate two AR(1) error terms with parameter rho = 0 (white noise) and rho = 0.9 respectively

- rho = 0:

>>> import numpy as np
>>> err1 = np.random.randn(100)
>>> ## generate regressor and dependent variable
>>> x = np.tile([-1, 1], 50)
>>> y1 = 1 + x + err1
>>> ## perform Durbin-Watson test
>>> dw_test('y1 ~ x', data=pd.DataFrame({'y1': y1, 'x': x}))

- rho = 0.9:

>>> from statsmodels.tsa.filters.filtertools import recursive_filter
>>> err2 = recursive_filter(err1, 0.9)
>>> y2 = 1 + x + err2
>>> dw_test('y2 ~ x', data=pd.DataFrame({'y2': y2, 'x': x}))
    """
    return _dw_test(input_obj, order_by, alternative, data)
