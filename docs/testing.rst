

The `testing` module
=====================

.. toctree::
   :maxdepth: 2

CheckModel Class
----------------

.. class:: CheckModel(fitted_model: RegressionResultsWrapper)

   Check Linear Regression Assumptions
   -----------------------------------

   The `CheckModel` class provides methods to test the assumptions of the linear regression model.
   These assumptions are:

   - Normality of residuals
   - Homoscedasticity (equal variance) of residuals
   - Independence of residuals
   - No Multicollinearity among predictors
   

   .. method:: __init__(fitted_model: RegressionResultsWrapper)

      Initializes the CheckModel class.

      :param fitted_model: The fitted linear regression model which is an instance of RegressionResultsWrapper.

   .. method:: check_normality(alpha=0.05, plot=True, return_pvals=False)

      Checks the normality assumption of the residuals using several statistical tests.

   .. method:: check_homocedasticity(alpha=0.05, plot=True, return_pvals=False)

      Checks the homoscedasticity assumption (equal variance of residuals) using several statistical tests.

   .. method:: check_independence(alpha=0.05, plot=True, return_vals=False)

      Checks the independence assumption of the residuals using several statistical tests.

   .. method:: check_multicollinearity(plot=True, return_cm=False)

      Checks the multicollinearity assumption among predictors using the variance inflation factor (VIF).

   .. method:: check_all(alpha=0.05, plot=True, return_vals=False)

      Checks all the assumptions of the linear regression model.

   Examples
   --------

   .. jupyter-execute::
      :hide-code:

      %config InlineBackend.figure_format = 'retina'

   .. jupyter-execute::

      import statsmodels.api as sm
      from sklearn.datasets import load_diabetes
      from estyp.testing import CheckModel

      diabetes = load_diabetes()
      X = diabetes["data"]
      y = diabetes["target"]
      X = sm.add_constant(X)
      model = sm.OLS(y, X)
      fitted_model = model.fit()
      cm = CheckModel(fitted_model)
      cm.check_all()


F Test to Compare Two Variances
-------------------------------

.. function:: var_test(x, y, ratio=1, alternative="two-sided", conf_level=0.95)

   Performs an F test to compare the variances of two samples from normal populations. This function is inspired by the `var.test()` function of the software R.

   :param x, y: numeric list, `np.array` or `pd.Series` of data values.
   :param ratio: the hypothesized ratio of the population variances of x and y.
   :param alternative: a character string specifying the alternative hypothesis, must be one of "two-sided" (default), "greater" or "less". You can specify just the initial letter.
   :param conf_level: a number between 0 and 1 indicating the confidence level of the interval.

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

   .. jupyter-execute::

      import numpy as np
      from estyp.testing import var_test

      np.random.seed(2023)
      x = np.random.normal(size=100)
      y = np.random.normal(size=100)

      print("1 - F Test for Two Samples")
      print(var_test(x, y))
      print("2 - F Test for Two Samples changing alternative hypothesis")
      print(var_test(x, y, alternative="less"))
      print("3 - F Test for Two Samples changing ratio")
      print(var_test(x, y, ratio=0.9, alternative="greater"))



Student's t-Test
----------------

.. function:: t_test(x, y=None, alternative="two-sided", mu=0, paired=False, var_equal=False, conf_level=0.95)

   Performs one and two sample t-tests on groups of data. This function is inspired by the `t.test()` function of the software R.

   :param x: a (non-empty) numeric container of data values.
   :param y: an (optional) numeric container of data values.
   :param alternative: a string specifying the alternative hypothesis, must be one of "two-sided" (default), "greater" or "less".
   :param mu: a number indicating the true value of the mean (or difference in means if you are performing a two sample test).
   :param paired: a logical indicating whether you want a paired t-test.
   :param var_equal: a logical variable indicating whether to treat the two variances as being equal. If `True` then the pooled variance is used to estimate the variance otherwise the Welch (or Satterthwaite) approximation to the degrees of freedom is used.
   :param conf_level: a number between 0 and 1 indicating the confidence level of the interval.

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

   .. jupyter-execute::

      import numpy as np
      from estyp.testing import t_test

      np.random.seed(2023)
      x = np.random.normal(size=100)
      y = np.random.normal(size=100)
      mu = 0.1

      print("1 - One Sample Test")
      print(t_test(x, mu=mu, alternative="less"))
      print("2 - Two Sample Test")
      print(t_test(x, y, mu=mu))
      print("3 - Two Sample Test with Equal Variances")
      print(t_test(x, y, mu=mu, var_equal=True, alternative="greater"))
      print("4 - Paired Test")
      print(t_test(x, y, mu=mu, paired=True))


Nested Models F-Test Function
-----------------------------

.. function:: nested_models_test(fitted_small_model, fitted_big_model)

   This function performs a nested models F-test using deviance from two fitted models from statsmodels library. The test compares two nested models: a larger or "big" model and a smaller or "small" model. The purpose of this test is to determine whether the larger model significantly improves the model fit compared to the smaller model by adding additional predictors.

   :param fitted_small_model: The fitted model representing the smaller/nested model. It has to come from statsmodels.
   :type fitted_small_model: RegressionResultsWrapper
   :param fitted_big_model: The fitted model representing the larger model, which includes all the predictors from the smaller model and potentially additional predictors. It has to come from statsmodels.
   :type fitted_big_model: RegressionResultsWrapper

   Returns:
   --------
   The function returns an object of class TestResults that contains the following information:

   - `method`: A string indicating the name of the statistical test (Nested models F-test).
   - `statistic`: The computed F-statistic value.
   - `estimate`: The difference in deviances between the models.
   - `df`: A dictionary with the degrees of freedom for the numerator and denominator of the F-statistic.
   - `p_value`: The p-value associated with the F-statistic.

   Examples:
   ---------

   - Example 1: With OLS

   .. jupyter-execute::

      import pandas as pd
      import statsmodels.api as sm
      from estyp.testing import nested_models_test

      data = pd.DataFrame({
          "x": [2.01, 2.99, 4.01, 5.01, 6.89],
          "y": [2, 3, 4, 5, 6]
      })
      model_small = sm.OLS.from_formula("y ~ 1", data).fit()
      model_big = sm.OLS.from_formula("y ~ x", data).fit()
      print(nested_models_test(model_small, model_big))

   - Example 2: With Logit

   .. jupyter-execute::

      data = pd.DataFrame({
          "x": [2.01, 2.99, 4.01, 3.01, 4.89],
          "y": [0, 1, 1, 0, 1]
      })
      model_small = sm.Logit.from_formula("y ~ 1", data).fit()
      model_big = sm.Logit.from_formula("y ~ x", data).fit()
      print(nested_models_test(model_small, model_big))



   - Example 3: With GLM

   .. jupyter-execute::

      data = pd.DataFrame({
          "x": [2.01, 2.99, 4.01, 5.01, 6.89],
          "y": [2, 3, 4, 5, 6]
      })
      model_small = sm.GLM.from_formula("y ~ 1", data, family = sm.families.Gamma()).fit()
      model_big = sm.GLM.from_formula("y ~ x", data, family = sm.families.Gamma()).fit()
      print(nested_models_test(model_small, model_big))


Test of Equal or Given Proportions
-----------------------------------

.. function:: prop_test(x, n=None, p=None, alternative="two-sided", conf_level=0.95, correct=True)

   :param x: A vector of counts of successes, a one-dimensional table with two entries, or a two-dimensional table (or matrix) with 2 columns, giving the counts of successes and failures, respectively.
   :type x: array_like
   :param n: A vector of counts of trials; ignored if x is a matrix or a table. If not provided, it is calculated as the sum of the elements in x.
   :type n: array_like, optional
   :param p: A vector of probabilities of success. The length of p must be the same as the number of groups specified by x, and its elements must be greater than 0 and less than 1.
   :type p: array_like, optional
   :param alternative: A character string specifying the alternative hypothesis, must be one of "two-sided" (default), "greater" or "less". You can specify just the initial letter. Only used for testing the null that a single proportion equals a given value, or that two proportions are equal; ignored otherwise.
   :type alternative: str, optional
   :param conf_level: Confidence level of the returned confidence interval. Must be a single number between 0 and 1. Only used when testing the null that a single proportion equals a given value, or that two proportions are equal; ignored otherwise.
   :type conf_level: float, optional
   :param correct: A logical indicating whether Yates' continuity correction should be applied where possible.
   :type correct: bool, optional

   Returns
   -------
   A data class with the following attributes:

   - `statistic`: float, The value of Pearson's chi-squared test statistic.
   - `df`: int, The degrees of freedom of the approximate chi-squared distribution of the test statistic.
   - `p_value`: float, The p-value of the test.
   - `estimate`: array_like, A vector with the sample proportions x/n.
   - `null_value`: float or array_like, The value of p if specified by the null hypothesis.
   - `conf_int`: array_like, A confidence interval for the true proportion if there is one group, or for the difference in proportions if there are 2 groups and p is not given, or None otherwise. In the cases where it is not None, the returned confidence interval has an asymptotic confidence level as specified by conf_level, and is appropriate to the specified alternative hypothesis.
   - `alternative`: str, A character string describing the alternative.
   - `method`: str, A character string indicating the method used, and whether Yates' continuity correction was applied.

   Examples
   --------

   .. jupyter-execute::

      import numpy as np
      from scipy import stats
      from estyp.testing import prop_test

      x = np.array([83, 90, 129, 70])
      n = np.array([86, 93, 136, 82])
      result = prop_test(x, n=n)
      print(result)


Test for Association/Correlation Between Paired Samples
-------------------------------------------------------

.. function:: cor_test(x, y, method="pearson", alternative="two-sided", conf_level=0.95, continuity=False)

   :param x: Numeric one-dimensional arrays, lists or pd.Series of data values. x and y must have the same length.
   :type x: array_like
   :param y: Numeric one-dimensional arrays, lists or pd.Series of data values. x and y must have the same length.
   :type y: array_like
   :param method: A string indicating which correlation coefficient is to be used for the test. One of "pearson", "kendall", or "spearman".
   :type method: str, optional
   :param alternative: Indicates the alternative hypothesis and must be one of "two-sided", "greater" or "less". "greater" corresponds to positive association, "less" to negative association.
   :type alternative: str, optional
   :param conf_level: Confidence level for the returned confidence interval. Currently only used for the Pearson product moment correlation coefficient if there are at least 4 complete pairs of observations.
   :type conf_level: float, optional
   :param continuity: If True, a continuity correction is used for Kendall's tau.
   :type continuity: bool, optional

   Returns
   -------
   A TestResults instance containing the following attributes:

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

   .. jupyter-execute::

      from sklearn import datasets
      from estyp.testing import cor_test

      iris = datasets.load_iris()
      sepal_length = iris.data[:, 0]
      petal_length = iris.data[:, 2]

      result = cor_test(sepal_length, petal_length, method="pearson")
      print(result)


Pearson's Chi-squared Test for Count Data
------------------------------------------

.. function:: chisq_test(x, y=None, p=None, correct=True, rescale_p=False)

   :param x: A numeric list or 2D list (matrix). x and y can also both be lists.
   :type x: array_like
   :param y: A numeric data; ignored if x is a matrix. If x is a list, y should be a list of the same length. The default is None.
   :type y: array_like, optional
   :param p: A list of probabilities of the same length as x. An error is raised if any entry of p is negative.
   :type p: array_like, optional
   :param correct: A boolean indicating whether to apply continuity correction when computing the test statistic for 2x2 tables: one half is subtracted from all abs(O-E) differences; however, the correction will not be bigger than the differences themselves. The default is True.
   :type correct: bool, optional
   :param rescale_p: A boolean; if True then p is rescaled (if necessary) to sum to 1. If rescale_p is False, and p does not sum to 1, an error is raised.
   :type rescale_p: bool, optional

   Returns
   -------
   A TestResults instance containing the following attributes:

   - `statistic`: The value of the chi-squared test statistic.
   - `df`: The degrees of freedom of the approximate chi-squared distribution of the test statistic.
   - `p_value`: The p-value for the test.
   - `method`: A string indicating the type of test performed, and whether continuity correction was used.
   - `expected`: The expected counts under the null hypothesis.

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

   Examples
   --------
   - Example 1: From Agresti(2007) p.39

   .. jupyter-execute::

      from estyp.testing import chisq_test

      M = [[762, 327, 468], [484, 239, 477]]
      result1 = chisq_test(M)
      print(result1)

   - Example 2: Effect of rescale_p

   .. jupyter-execute::

      x = [12, 5, 7, 7]
      p = [0.4, 0.4, 0.2, 0.2]
      result2 = chisq_test(x, p=p, rescale_p=True)
      print(result2)

   - Example 3.1: Testing for population probabilities

   .. jupyter-execute::

      x = [20, 15, 25]
      result31 = chisq_test(x)
      print(result31)

   - Example 3.2: A second example of testing for population probabilities

   .. jupyter-execute::

      x = [89,37,30,28,2]
      p = [0.40,0.20,0.20,0.19,0.01]
      result32 = chisq_test(x, p=p)
      print(result32)

   - Example 4: Goodness of fit

   .. jupyter-execute::

      x = [1, 2, 3, 4, 5, 6]
      y = [6, 1, 2, 3, 4, 5]
      result4 = chisq_test(x, y=y)
      print(result4)


Durbin-Watson Test for Autocorrelation
--------------------------------------

.. function:: dw_test(input_obj, order_by=None, alternative="greater", data=None)

    Performs the Durbin-Watson test for autocorrelation of disturbances. Includes an approximated p-value taken from the `lmtest::dwtest()` function in R.

   :param input_obj: A formula description for the model to be tested (or a OLS object or a fitted OLS object).
   :type input_obj: str, OLS, RegressionResultsWrapper
   :param order_by: The observations in the model are ordered by the order specified in this parameter. If set to None (the default) the observations are assumed to be ordered (e.g., a time series).
   :type order_by: Series, np.ndarray, optional
   :param alternative: A character string specifying the alternative hypothesis. One of “greater” (default), “two-sided”, or “less”.
   :type alternative: str, optional
   :param data: An optional pandas DataFrame containing the variables in the model.
   :type data: pd.DataFrame, optional

    The Durbin-Watson test has the null hypothesis that the autocorrelation of the disturbances is 0. 
    It is possible to test against the alternative that it is greater than, not equal to, or less than 0, respectively. 
    This can be specified by the alternative parameter.

    For large sample sizes the algorithm might fail to compute the p value; in that case a warning is printed 
    and an approximate p value will be given; this p value is computed using a normal approximation 
    with mean and variance of the Durbin-Watson test statistic.

    :returns: A `TestResults` instance containing:

        - `statistic`: the test statistic.
        - `method`: a character string with the method used.
        - `alternative`: a character string describing the alternative hypothesis.
        - `p_value`: the corresponding p-value.
        - `estimate`: the estimate of the autocorrelation of firt order of the residuals obtained from DW statistic.

    References
    ----------

   - J. Durbin & G.S. Watson (1950), Testing for Serial Correlation in Least Squares Regression I. Biometrika 37, 409–428.
   - J. Durbin & G.S. Watson (1951), Testing for Serial Correlation in Least Squares Regression II. Biometrika 38, 159–177.
   - J. Durbin & G.S. Watson (1971), Testing for Serial Correlation in Least Squares Regression III. Biometrika 58, 1–19.
   - R.W. Farebrother (1980), Pan's Procedure for the Tail Probabilities of the Durbin-Watson Statistic. Applied Statistics 29, 224–227.
   - W. Krämer & H. Sonnberger (1986), The Linear Regression Model under Test. Heidelberg: Physica.

    Examples
    --------

    Generate two AR(1) error terms with parameter rho = 0 (white noise) and rho = 0.9 respectively

    - rho = 0:

    .. jupyter-execute::

      import numpy as np
      import statsmodels.api as sm
      from estyp.testing import dw_test

      np.random.seed(123)
      err1 = np.random.randn(100)
      ## generate regressor and dependent variable
      x = np.tile([-1, 1], 50)
      y1 = 1 + x + err1
      data = pd.DataFrame({'y1': y1, 'x': x})
      ## perform Durbin-Watson test
      model = sm.OLS.from_formula('y1 ~ x', data=data)
      print(dw_test(model))

    - rho = 0.9:

    .. jupyter-execute::

      from statsmodels.tsa.filters.filtertools import recursive_filter
      err2 = recursive_filter(err1, 0.9)
      y2 = 1 + x + err2
      print(dw_test('y2 ~ x', data=pd.DataFrame({'y2': y2, 'x': x})))

   More examples:

   .. jupyter-execute::

      from estyp.testing import dw_test

      x = np.linspace(0, 1, 120)

      print("Example 1")
      y1 = 2 * x + np.tile([0.2, -0.2, -0.1, 0.1, 0.05, -0.07], 20)
      model1 = sm.OLS(y1, sm.add_constant(x))
      result_example_1 = dw_test(model1, alternative="less")
      print(result_example_1)

      print("Example 2")
      y2 = 2 * x + np.tile([0.2, -0.2, -0.1, 0.1], 30)
      model2 = sm.OLS(y2, sm.add_constant(x)).fit()
      result_example_2 = dw_test(model2, alternative="two-sided")
      print(result_example_2)

      print("Example 3")
      np.random.seed(123)
      y3 = np.random.normal(size=120)
      df_example_3 = pd.DataFrame({"x": x, "y": y3})
      result_example_3 = dw_test("y ~ x", data=df_example_3, alternative="greater")
      print(result_example_3)


