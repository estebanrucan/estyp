from typing import Literal
from statsmodels.regression.linear_model import RegressionResultsWrapper
from estyp.testing.__base import TestResults, __var_test, __t_test, __nested_models_test
from pandas import Series

def var_test(
    x,
    y,
    ratio=1,
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
    x,
    y=None,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    mu:float    = 0,
    paired      = False,
    var_equal   = False,
    conf_level  = 0.95
):
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
    fitted_big_model: RegressionResultsWrapper
):
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