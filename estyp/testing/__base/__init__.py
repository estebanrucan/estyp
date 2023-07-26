import numpy as np
from pandas import Series
from scipy.stats import f as fisher
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel

from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.genmod.families.family import Gaussian


class TestResults:
    def __init__(self, res: dict, names: dict):
        self.__dict__.update(res)
        self.__names = names

    def __repr__(self):
        # p-value
        p_value = self.p_value
        if p_value < 0.0001:
            p_value = "<0.0001"
        else:
            p_value = f"{p_value:0.4f}"
        # df
        df = self.df
        if isinstance(df, (float, int)):
            if df == float(int(df)):
                df = int(df)
        elif isinstance(df, dict):
            df = {k: f"{v:0.2f}" for k, v in df.items()}
        elif isinstance(df, float):
            df = f"{df:0.2f}"
        # estimates
        estimate = self.estimate
        if isinstance(estimate, list):
            estimate = [float(f"{e:0.6f}") for e in estimate]
        elif estimate == float(int(estimate)):
            estimate = int(estimate)
        elif isinstance(estimate, float):
            estimate = f"{estimate:0.6f}"

        string = f"""
    {self.method}
    {len(self.method) * "-"}
    {self.__names['statistic']} = {self.statistic:0.4f} | df: {df} | p-value = {p_value}
    alternative hypothesis: {self.__names["alternative"]}"""
        if self.__dict__.get("conf_int"):
            cl = self.conf_level * 100
            if cl == float(int(cl)):
                cl = int(cl)
            string += f"""
    {cl} percent confidence interval:
    {" "}{self.conf_int[0]:0.6f} {self.conf_int[1]:0.6f}"""
        string += f"""
    sample estimates:
    {" " * 2}{self.__names["estimate"]}: {estimate}
    """
        return string


def __var_test(x, y, ratio, alternative, conf_level):
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


def __t_test(x, y, alternative, mu, paired, var_equal, conf_level):
    if isinstance(conf_level, float):
        if conf_level <= 0 or conf_level >= 1:
            raise ValueError("'conf_level' must be between 0 and 1")
    else:
        raise TypeError("'conf_level' must be a single number between 0 and 1")

    if isinstance(alternative, str):
        if alternative not in ["two-sided", "less", "greater"]:
            raise ValueError(
                "'alternative' must be one of 'two-sided', 'less', or 'greater'"
            )
    else:
        raise TypeError("'alternative' must be a string")

    if paired:
        if y is None:
            raise ValueError("If 'paired' is True, 'y' must be provided")
        else:
            result = ttest_rel(x, y + mu, alternative=alternative)
    else:
        if y is None:
            result = ttest_1samp(x, mu, alternative=alternative)
        else:
            result = ttest_ind(x, y + mu, alternative=alternative, equal_var=var_equal)

    lower, upper = result.confidence_interval(conf_level)
    ci = (lower, upper)
    t, p = result.statistic, result.pvalue
    df = result.df

    alternative_dire = (
        "not equal to" if alternative == "two-sided" else f"{alternative} than"
    )
    alternative_conc = "mean difference" if paired else "difference in means"

    if y is None:
        est = x.mean()
        estimate = "mean of x"
        alternative_name = f"true mean is {alternative_dire} {mu}"
        method = "One Sample t-test"
    else:
        welch = "Welch's " if (not var_equal and not paired) else ""
        title = "Paired" if paired else "Two Sample"
        est = [x.mean(), y.mean()]
        estimate = "[mean of x, mean of y]"
        alternative_name = f"true {alternative_conc} is {alternative_dire} {mu}"
        method = f"{welch}{title} t-test"

    names = {
        "statistic": "T",
        "estimate": estimate,
        "alternative": alternative_name,
    }

    res = {
        "method": method,
        "statistic": t,
        "estimate": est,
        "df": df,
        "p_value": p,
        "conf_int": ci,
        "mu": mu,
        "conf_level": conf_level,
        "alternative": alternative,
    }

    return TestResults(res, names)


def __deviance(fitted_model, response):
    if isinstance(fitted_model, RegressionResultsWrapper):
        return Gaussian().deviance(endog=response, mu=fitted_model.fittedvalues)
    return fitted_model.deviance


def __nested_models_test(fitted_small_model, fitted_big_model, response):
    n = fitted_big_model.fittedvalues.shape[0]
    p_big = fitted_big_model.params.shape[0] - 1
    p_small = fitted_small_model.params.shape[0] - 1
    d_big = __deviance(fitted_big_model, response)
    d_small = __deviance(fitted_small_model, response)
    df_num = p_big - p_small
    df_den = n - p_big - 1
    f_num = (d_small - d_big) / df_num
    f_den = d_big / df_den
    f_stat = f_num / f_den
    p_value = fisher(dfn=df_num, dfd=df_den).sf(f_stat)

    names = {
        "statistic": "F",
        "estimate": "Difference in deviances between models",
        "alternative": "big model is true",
    }

    res = {
        "method": "Nested models F-test",
        "statistic": f_stat,
        "estimate": d_small - d_big,
        "df": {"df_num": df_num, "df_den": df_den},
        "p_value": p_value,
    }

    return TestResults(res, names)
