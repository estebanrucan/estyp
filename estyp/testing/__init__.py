import numpy as np
from pandas import Series
from scipy.stats import f as fisher
from typing import Literal
from __base import TestResults


def var_test(
    x,
    y,
    ratio=1,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    conf_level=0.95,
) -> TestResults:
    """
# F-ratio Test

Performs an F-ratio test to compare the variances of two samples.

## Arguments

* `x`: The first sample.
* `y`: The second sample.
* `ratio`: The ratio of the two variances under the null hypothesis.
* `alternative`: The alternative hypothesis. Can be one of "two-sided", "less", or "greater".
* `conf_level`: The confidence level for the confidence interval.

## Returns

A `TestResults` instance containing the following attributes:

* `statistic`: The test statistic.
* `p_value`: The p-value.
* `ci`: The confidence interval for the ratio of variances.
* `estimate`: The estimated ratio of variances.
* `alternative`: The alternative hypothesis.

## Example

```python
import numpy as np
x = np.random.normal(size=100)
y = np.random.normal(size=100)

res = var_test(x, y, alternative="two-sided", conf_level=0.9)

print(res)
    """

    if not (isinstance(ratio, float) or isinstance(ratio, int)):
        raise ValueError("'ratio' must be a single positive number")
    else:
        if not ratio > 0:
            raise ValueError("'ratio' must be a single positive number")

    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(
            "'alternative' must be one of 'two.sided', 'less', or 'greater'"
        )

    if not (isinstance(conf_level, float)):
        raise ValueError("'conf_level' must be a single number between 0 and 1")
    else:
        if not (conf_level > 0 and conf_level < 1):
            raise ValueError("'conf_level' must be a single number between 0 and 1")

    if isinstance(x, np.ndarray) or isinstance(x, Series):
        df_x = x.shape[0] - 1
        v_x = np.var(x)
    else:
        df_x = len(x) - 1
        v_x = np.var(x)

    if isinstance(y, np.ndarray) or isinstance(y, Series):
        df_y = y.shape[0] - 1
        v_y = np.var(y)
    else:
        df_y = len(y) - 1
        v_y = np.var(y)

    est = v_x / v_y
    stat = est / ratio

    f = fisher(dfn=df_x, dfd=df_y)
    pval = f.cdf(stat)

    if alternative == "two-sided":
        pval = 2 * min([pval, 1 - pval])
        beta = (1 - conf_level) / 2
        ci = [est / f.ppf(1 - beta), est / f.ppf(beta)]
    elif alternative == "greater":
        pval = 1 - pval
        ci = [est / f.ppf(conf_level), np.inf]
    else:
        ci = [0, est / f.ppf(1 - conf_level)]

    alternative_name = (
        "not equal to" if alternative == "two-sided" else f"{alternative} than"
    )

    names = {
        "statistic": "F",
        "estimate": "ratio of variances",
        "alternative": f"true ratio of variances is {alternative_name} {ratio}",
    }

    res = {
        "method": "F test to compare two variances",
        "statistic": stat,
        "estimate": est,
        "df": {"x": df_x, "y": df_y},
        "p_value": pval,
        "conf_int": ci,
        "conf_level": conf_level,
        "alternative": alternative,
    }

    return TestResults(res, names)
