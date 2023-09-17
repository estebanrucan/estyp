import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from pandas import DataFrame, Series, crosstab
from scipy.linalg import solve_triangular
from scipy.stats import chi2
from scipy.stats import f as fisher
from scipy.stats import (kstest, norm, probplot, shapiro, ttest_1samp,
                         ttest_ind, ttest_rel)
from statsmodels.api import GLM, OLS
from statsmodels.genmod.families.family import Gaussian
from statsmodels.stats.diagnostic import (acorr_breusch_godfrey,
                                          acorr_ljungbox, het_breuschpagan,
                                          het_goldfeldquandt, het_white)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import (jarque_bera,
                                         omni_normtest)

class bcolors:
    HEADER = "\033[95m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"



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
        if self.__dict__.get("df") is not None:
            df = self.df
            if isinstance(df, (float, int)):
                if df == float(int(df)):
                    df = int(df)
                else:
                    df = f"{df:0.2f}"
            elif isinstance(df, dict):
                df = {k: round(v, 2) for k, v in df.items()}
            elif isinstance(df, float):
                df = f"{df:0.2f}"
        else:
            df = None
        # estimates
        if self.__dict__.get("estimate") is not None:
            estimate = self.estimate
            if isinstance(estimate, (list, np.ndarray)):
                estimate = [float(f"{e:0.6f}") for e in estimate]
            elif estimate == float(int(estimate)):
                estimate = int(estimate)
            elif isinstance(estimate, float):
                estimate = f"{estimate:0.6f}"

        string = f"""
    {bcolors.BOLD + bcolors.UNDERLINE + self.method + bcolors.ENDC}
    {self.__names['statistic']} = {self.statistic:0.4f} |{' df = ' + str(df) + ' |' if df is not None else ''} p-value = {p_value}
    alternative hypothesis: {self.__names["alternative"]}"""
        if self.__dict__.get("conf_int") is not None:
            if self.conf_int is not None:
                cl = self.conf_level * 100
                if cl == float(int(cl)):
                    cl = int(cl)
                string += f"""
    {cl} percent confidence interval:
    {" "}{self.conf_int[0]:0.6f} {self.conf_int[1]:0.6f}"""
        if self.__dict__.get("estimate") is not None:
            string += f"""
    sample estimates:
    {" " * 2}{self.__names["estimate"]}: {estimate}
    """
        return string

def _dw_test(input_obj, order_by, alternative, data):

    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("Invalid alternative. Must be one of 'two-sided', 'greater', or 'less'")

    if data is not None and not isinstance(data, DataFrame):
        raise ValueError("data must be a pandas DataFrame")

    if isinstance(input_obj, str):
        if "~" not in input_obj:
            raise ValueError("if str, input_obj must be written as formula syntax.")
        formula = input_obj
        dependent_var = formula.split("~")[0].strip()
        independent_vars = [var.strip() for var in formula.split("~")[1].split("+")]
        y = data[dependent_var].values
        X = sm.add_constant(data[independent_vars]).values
        model = sm.OLS(y, X).fit()
        res = model.resid
    elif isinstance(input_obj, sm.OLS):
        model = input_obj
        X = model.exog
        y = model.endog
        model = model.fit()
        res = model.resid
    elif isinstance(input_obj, sm.regression.linear_model.RegressionResultsWrapper):
        model = input_obj
        X = model.model.exog
        y = model.model.endog
        model = input_obj
        res = model.resid
    else:
        try:
            X = model.model.exog
            y = model.model.endog
            model = input_obj
            res = model.resid
        except:
            raise ValueError("Not implemented yet.")
    
    if order_by:
        order_index = data[order_by].argsort().values
        X = X[order_index]
        y = y[order_index]
        
    n, k = X.shape

    dw = np.sum(np.diff(res)**2) / np.sum(res**2)
    r = - (dw / 2 - 1)
    
    Q1 = solve_triangular(np.linalg.qr(X)[1], np.eye(k), lower=False)
    
    if n < 3:
        pval = 1
        warnings.warn("sample size is too small to perform Durbin-Watson test")
    else:
        if n < max(5, k):
            pval = 1
            warnings.warn("sample size is too small to perform Durbin-Watson test")
        else:
            AX = np.diff(X, axis=0)
            AX = np.vstack([X[0] - X[1], AX])
            AX[-1] += X[-1] - X[-2]
            XAXQ = X.T @ AX @ Q1
            P = 2 * (n - 1) - np.trace(XAXQ)
            Q = 2 * (3 * n - 4) - 2 * np.trace(AX.T @ AX @ Q1) + np.trace(XAXQ @ XAXQ)
            dmean = P / (n - k)
            dvar = 2 / ((n - k) * (n - k + 2)) * (Q - P * dmean)
            if alternative == "two-sided":
                pval = 2 * norm.sf(np.abs(dw - dmean), scale=np.sqrt(dvar))
            elif alternative == "less":
                pval = norm.sf(dw, loc=dmean, scale=np.sqrt(dvar))
            elif alternative == "greater":
                pval = norm.cdf(dw, loc=dmean, scale=np.sqrt(dvar))
    
    direction = "not" if alternative == "two-sided" else f"{alternative} than"

    names = {
        "statistic": "DW",
        "estimate": "r",
        "alternative": f"true autocorrelation is {direction} 0"
    }

    res = {
        "method": "Durbin-Watson test",
        "statistic": dw,
        "estimate": r,
        "p_value": pval,
        "alternative": alternative,
    }

    return TestResults(res, names)


class _CheckNormality:
    def __init__(self, fitted_model):
        self.model = fitted_model
        self.__e = fitted_model.resid
        self.__scale = fitted_model.scale ** (1 / 2)
        self.__norm = norm(loc=0, scale=self.__scale)

    def plot(self):
        xn = np.linspace(self.__e.min() * 1.3, self.__e.max() * 1.3, 100)
        yn = self.__norm.pdf(xn)
        x, y = probplot(self.__e, fit=False, dist=self.__norm)
        minimo = self.__e.min() * 1.2
        maximo = self.__e.max() * 1.2

        # Graficos
        # G1
        fig, ax = plt.subplots(figsize=(12, 4), ncols=2, nrows=1)
        ax[0].hist(self.__e, bins=20, edgecolor="white", density=True, alpha=0.9)
        ax[0].plot(xn, yn, color="darkgreen")
        ax[0].fill_between(xn, 0, yn, color="darkgreen", alpha=0.3)
        # G2
        ax[1].scatter(x, y, alpha=0.8)
        ax[1].plot((-1e10, 1e10), (-1e10, 1e10), color="darkgreen", linewidth=2)
        ax[1].set_xlim(minimo, maximo)
        ax[1].set_ylim(minimo, maximo)

        # Etiquetas
        fig.suptitle("Normality of residuals:")
        # G1
        ax[0].set_xlabel("Residuals")
        ax[0].set_ylabel("Density")
        ax[0].set_title("Histogram should be similar to the curve", fontsize=10)
        # G2
        ax[1].set_ylabel("Residuals Quantiles")
        ax[1].set_xlabel("Theoretical Quantiles")
        ax[1].set_title("Points should be close to the line", fontsize=10)
        return fig, ax

    def test(self, sign_level=0.05):
        _, pval_ks = kstest(self.__e, cdf=self.__norm.cdf)
        _, pval_sh = shapiro(self.__e)
        _, pval_jb, *_ = jarque_bera(self.__e)
        _, pval_om = omni_normtest(self.__e)

        text_ok = (
            lambda name, p: f"- Residuals appear as normally distributed according to {name} test (p-value = {p:0.3f})."
        )
        test_no = (
            lambda name, p: f"- Residuals don't appear as normally distributed according to {name} test (p-value = {p:0.3f})."
        )
        names = ["KS", "Shapiro-Wilk", "Jarque-Bera", "Omni"]
        pvals = [pval_ks, pval_sh, pval_jb, pval_om]

        print(
            bcolors.BOLD + bcolors.UNDERLINE + "Normality tests results:" + bcolors.ENDC
        )

        for name, pval in zip(names, pvals):
            if pval >= sign_level:
                print(bcolors.OKGREEN + text_ok(name, pval) + bcolors.ENDC)
            else:
                print(bcolors.FAIL + test_no(name, pval) + bcolors.ENDC)
        return dict(zip(names, pvals))


class _CheckHomocedasticity:
    def __init__(self, fitted_model):
        self.model = fitted_model
        self.__e = fitted_model.resid

    def plot(self):
        std_resid = self.model.get_influence().resid_studentized_internal
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(self.model.fittedvalues, std_resid, alpha=0.8)
        ax.axhline(0, color="darkgreen", linestyle="--")
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Studentized Residuals")
        ax.set_title(
            "Homocedasticity of residuals:\nIt shouldn't be any pattern in the plot",
            fontsize=10,
        )
        return fig, ax

    def test(self, sign_level=0.05):
        *_, pval_bp = het_breuschpagan(self.__e, self.model.model.exog)
        *_, pval_w = het_white(self.__e, self.model.model.exog)
        _, pval_gq, _ = het_goldfeldquandt(self.__e, self.model.model.exog)
        text_ok = (
            lambda name, p: f"- Error variance appears to be homoscedastic according to {name} test (p-value = {p:0.3f})."
        )
        test_no = (
            lambda name, p: f"- Heteroscedasticity (non-constant error variance) detected according to {name} test (p-value = {p:0.3f})."
        )
        names = ["Breusch-Pagan", "White", "Goldfeld-Quandt"]
        pvals = [pval_bp, pval_w, pval_gq]
        print(
            bcolors.BOLD
            + bcolors.UNDERLINE
            + "Homocedasticity tests results:"
            + bcolors.ENDC
        )

        for name, pval in zip(names, pvals):
            if pval >= sign_level:
                print(bcolors.OKGREEN + text_ok(name, pval) + bcolors.ENDC)
            else:
                print(bcolors.FAIL + test_no(name, pval) + bcolors.ENDC)
        return dict(zip(names, pvals))


class _CheckIndependence:
    def __init__(self, fitted_model):
        self.model = fitted_model
        self.__e = fitted_model.resid

    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        sm.graphics.tsa.plot_acf(self.__e, lags=20, ax=ax)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(
            "Independence of residuals:\nAll bars should be within the blue region",
            fontsize=10,
        )
        return fig, ax

    def test(self, sign_level=0.05):
        pval_dw = _dw_test(self.model, None, "two-sided", None).p_value
        pval_bp = acorr_ljungbox(self.__e, boxpierce=True, auto_lag=True)["bp_pvalue"].min()
        *_, pval_bg = acorr_breusch_godfrey(self.model)
        text_ok = (
            lambda name, p: f"- Residuals appear to be independent and not autocorrelated according to {name} test (p-value = {p:0.3f})."
        )
        test_no = (
            lambda name, p: f"- Autocorrelated residuals detected according to {name} test (p-value = {p:0.3f})."
        )
        names = ["Durbin-Watson", "Box-Pierce", "Breusch-Godfrey"]
        pvals = [pval_dw, pval_bp, pval_bg]

        print(
            bcolors.BOLD
            + bcolors.UNDERLINE
            + "Independence tests results:"
            + bcolors.ENDC
        )

        for name, pval in zip(names, pvals):
            if pval >= sign_level:
                print(bcolors.OKGREEN + text_ok(name, pval) + bcolors.ENDC)
            else:
                print(bcolors.FAIL + test_no(name, pval) + bcolors.ENDC)
        return dict(zip(names, pvals))


class _CheckMulticollinearity:
    def __init__(self, fitted_model):
        self.fitted_model = fitted_model
        self.exog_df = DataFrame(
            fitted_model.model.exog, columns=fitted_model.model.exog_names
        )

    def test(self):
        cn = self.fitted_model.condition_number
        print(
            bcolors.BOLD
            + bcolors.UNDERLINE
            + "Multicollinearity test results:"
            + bcolors.ENDC
        )
        if cn < 20:
            print(
                bcolors.OKGREEN
                + f"The model is unlikely to have multicollinearity problems (condition number = {cn:,.2f})."
                + bcolors.ENDC
            )
        else:
            print(
                bcolors.FAIL
                + f"- The model may have multicollinearity problems (condition number = {cn:,.2f})."
                + bcolors.ENDC
            )

    def plot(self):
        # Calcula el VIF para cada predictor
        vif = DataFrame()
        vif["VIF Factor"] = [
            variance_inflation_factor(self.exog_df.values, i)
            for i in range(self.exog_df.shape[1])
        ]
        vif["features"] = self.exog_df.columns

        # Ordena el dataframe por el VIF
        vif = vif.sort_values(by="VIF Factor")

        # Dibuja el gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [
            "dodgerblue" if 0 < x <= 5 else ("green" if 5 < x < 10 else "red")
            for x in vif["VIF Factor"]
        ]
        ax.hlines(y=vif["features"], xmin=0, xmax=vif["VIF Factor"], color=colors)
        for x, y, color in zip(vif["VIF Factor"], vif["features"], colors):
            ax.plot(x, y, "o", color=color)
        for x, y, tex in zip(vif["VIF Factor"], vif["features"], vif["VIF Factor"]):
            t = ax.text(
                x=x + 0.05 if max(vif["VIF Factor"]) < 7 else x + 1,
                y=y,
                s=round(tex, 2),
                horizontalalignment="left",
                verticalalignment="center",
                fontdict={
                    "color": "dodgerblue"
                    if 0 < x <= 5
                    else ("green" if 5 < x < 10 else "red"),
                    "size": 14,
                },
            )

        ax.fill_betweenx(vif["features"], vif["VIF Factor"], color="gray", alpha=0.1)

        ax.set_xlabel("VIF Factor")
        ax.set_title(
            "Variance Inflation Factor (VIF):\nAll bars should be below 10", size=10
        )
        ax.spines[["right", "top"]].set_visible(False)
        return fig, ax

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
    try:
        fitted_model.model
        ok = True
    except:
        ok = False
    if ok:
        if isinstance(fitted_model.model, OLS):
            return Gaussian().deviance(endog=response, mu=fitted_model.fittedvalues)
        elif isinstance(fitted_model.model, GLM):
            return fitted_model.deviance
        else:
            return sum(fitted_model.resid_dev ** 2)
    else:
        return fitted_model.deviance


def __nested_models_test(fitted_small_model, fitted_big_model):
    try:
        response = fitted_big_model.model.data.endog
    except:
        response = fitted_big_model.y
        
    n = response.shape[0]
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
    diff_dev = float(d_small - d_big)
    names = {
        "statistic": "F",
        "estimate": "Difference in deviances between models",
        "alternative": "bigger model is true",
    }

    res = {
        "method": "Nested models F-test",
        "statistic": f_stat,
        "estimate": diff_dev,
        "df": {"df_num": df_num, "df_den": df_den},
        "p_value": p_value,
    }

    return TestResults(res, names)


def __prop_test(x, n, p, alternative, conf_level, correct):
    p_status = p is not None
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, (Series, DataFrame)):
        x = x.values
    if isinstance(x, (int, float)):
        x = np.array([x])
   
    if n is None and len(x.shape) == 1:
        if x.shape[0] != 2:
            raise ValueError("'x' should have 2 entries")
        l = 1
        n = np.sum(x)
        x = np.array(x[0])
    elif len(x.shape) == 2:  # If x is a matrix
        if x.shape[1] != 2:
            raise ValueError("'x' must have 2 columns")
        l = x.shape[0]
        n = np.sum(x, axis=1)
        x = x[:, 0]
    elif n is not None and not isinstance(n, int) and len(x) != len(n):
        l = x.shape[0]
        raise ValueError("'x' and 'n' must have the same length")

    # Ensure finite data
    x = np.array(x)[np.isfinite(x)]
    n = np.array(n)[np.isfinite(n)]

    k = x.shape[0]
    if k < 1:
        raise ValueError("Not enough data")

    if any(n <= 0):
        raise ValueError("Elements of 'n' must be positive")

    if any(x < 0):
        raise ValueError("Elements of 'x' must be nonnegative")

    if any(x > n):
        raise ValueError("Elements of 'x' must not be greater than those of 'n'")

    if p is None and k == 1:
        p = np.array([0.5])
    
    if p is not None:
        if isinstance(p, list):
            p = np.array(p)
        else:
            p = np.array([p])

        if len(p) != k:
            raise ValueError("'p' must have the same length as 'x' and 'n'")

        if any(np.array(p) <= 0) or any(np.array(p) >= 1):
            raise ValueError("Elements of 'p' must be in (0,1)")

    if k > 2 or (k == 2 and p is not None):
        alternative = "two-sided"

    if not (0 < conf_level < 1):
        raise ValueError("'conf_level' must be a single number between 0 and 1")

    if isinstance(n, (list, Series)):
        n = np.array(n)

    estimate = x / n
    yates = 0.5 if correct and k <= 2 else 0
    if k == 1:
        z = norm.ppf((1 + conf_level) / 2) if alternative == "two-sided" else norm.ppf(conf_level)
        yates = min(yates, abs(x[0] - n * p))
        z22n = z**2 / (2 * n)
        p_c = estimate[0] + yates / n
        p_u = 1 if p_c >= 1 else (p_c + z22n + z * np.sqrt(p_c * (1 - p_c) / n + z22n / (2 * n))) / (1 + 2 * z22n)
        p_c = estimate[0] - yates / n
        p_l = 0 if p_c <= 0 else (p_c + z22n - z * np.sqrt(p_c * (1 - p_c) / n + z22n / (2 * n))) / (1 + 2 * z22n)
        conf_int = {
            "two-sided": (max(p_l.item(), 0), min(p_u.item(), 1)),
            "greater": (max(p_l.item(), 0), 1),
            "less": (0, min(p_u.item(), 1))
        }[alternative]
        conf_int = (conf_int[0], conf_int[1])
    elif k == 2 and p is None:
        delta = estimate[0] - estimate[1]
        yates = min(yates, abs(delta) / np.sum(1 / n))
        width = (norm.ppf((1 + conf_level) / 2) if alternative == "two-sided" else norm.ppf(conf_level)) * \
                np.sqrt(np.sum(estimate * (1 - estimate) / n)) + yates * np.sum(1 / n)
        conf_int = {
            "two-sided": (max(delta - width, -1), min(delta + width, 1)),
            "greater": (max(delta - width, -1), 1),
            "less": (-1, min(delta + width, 1))
        }[alternative]
    else:
        conf_int = None

    if p is None:
        p = np.sum(x) / np.sum(n)
        parameter = k - 1
    else:
        parameter = k
    e = np.vstack([n * p, n * (1 - p)]).T
    X = np.vstack([x, n - x]).T
    chi2_stat = np.sum((np.abs(X - e) - yates)**2 / e)

    if alternative == "two-sided":
        p_value = 1 - chi2.cdf(chi2_stat, parameter)
    else:
        z = np.sign(estimate - p) * np.sqrt(chi2_stat)
        if isinstance(z, np.ndarray):
            if z.shape[0] > 1:
                z = z[0]
        p_value = 1 - norm.cdf(z) if alternative == "greater" else norm.cdf(z)

    
    method = f"{k}-sample test for {'equality of' if p_status else 'given'} proportions {'with' if yates else 'without'} continuity correction"

    
    direction = "not equal to" if alternative == "two-sided" else f"{alternative} than"
    if not p_status:  # if p is None
        if k == 1:
            alternative_name = f"the true proportion is {direction} 0.5"
        else:
            alternative_name = "the true proportions are not all equal"
    else:  # if p is not None
        if k == 1:
            alternative_name = f"the true proportion is {direction} {p[0]:0.4f}"
        else:
            alternative_name = f"the true proportions are {direction} {p:0.4f}"
    
    names = {
        "statistic": "X-squared",
        "estimate": "proportion(s)",
        "alternative": alternative_name,
    }
    res = {
        "method": method,
        "statistic": chi2_stat,
        "estimate": estimate.item() if k == 1 else estimate,
        "df": parameter,
        "p_value": p_value.item() if alternative != "two-sided" else p_value,
        "conf_int": conf_int,
        "null_value": p.item() if len(p.shape) != 0 and p.shape[0] == 1 else p,
        "conf_level": conf_level,
        "alternative": alternative,
    }
    return TestResults(res, names)


def __cor_test(x, y, method, alternative, conf_level, continuity):
    
    if not isinstance(x, (list, np.ndarray, Series)):
        raise TypeError("'x' must be a list, numpy array, or pandas Series")
    if isinstance(x, (Series, np.ndarray)):
        if len(x.shape) != 1:
            raise ValueError("'x' must be one-dimensional")
    if isinstance(y, (Series, np.ndarray)):
        if len(y.shape) != 1:
            raise ValueError("'x' must be one-dimensional")
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must have the same length")
    
    res = {
        "conf_level": conf_level,
        "alternative": alternative,
        "null_value": 0
    }

    
    direction = "not equal to" if alternative == "two-sided" else f"{alternative} than"
    
    n = len(x)
    
    if method == "pearson":
        if n < 3:
            raise ValueError("not enough finite observations")
        
        r = np.corrcoef(x, y)[0, 1]
        df = n - 2
        
        t_statistic = np.sqrt(df) * r / np.sqrt(1 - r ** 2)
        
        # Confidence interval calculation following R's logic
        if n > 3:
            z = np.arctanh(r)
            sigma = 1 / np.sqrt(n - 3)
            
            if alternative == "less":
                lo_z, hi_z = -np.inf, z + sigma * stats.norm.ppf(conf_level)
            elif alternative == "greater":
                lo_z, hi_z = z - sigma * stats.norm.ppf(conf_level), np.inf
            else:  # two-sided
                z_crit = sigma * stats.norm.ppf((1 + conf_level) / 2)
                lo_z, hi_z = z - z_crit, z + z_crit
                
            lo, hi = np.tanh((lo_z, hi_z))
        else:
            lo, hi = None, None
        
        # P-value
        if alternative == "less":
            p_value = stats.t.cdf(t_statistic, df)
        elif alternative == "greater":
            p_value = 1 - stats.t.cdf(t_statistic, df)
        else:  # two-sided
            p_value = 2 * min(stats.t.cdf(t_statistic, df), 1 - stats.t.cdf(t_statistic, df))
        
        names = {
            "statistic": "t",
            "estimate": "cor",
            "alternative": f"true correlation is {direction} 0"
        }
        
        res.update({
            "method": "Pearson's product-moment correlation",
            "estimate": r,
            "statistic": t_statistic,
            "df": df,
            "p_value": p_value,
            "conf_int": (lo, hi)
        })
        
        return TestResults(res, names)
    
    elif method == "spearman":
        r, p_value = stats.spearmanr(x, y, alternative=alternative)
        
        # Calculate S statistic
        S = (n ** 3 - n) * (1 - r)/6
        
        names = {
            "statistic": "S",
            "estimate": "rho",
            "alternative": f"true rho is {direction} 0"
        }
        
        res.update({
            "method": "Spearman's rank correlation rho",
            "estimate": r,
            "statistic": S,
            "p_value": p_value
        })
        
        return TestResults(res, names)
    
    elif method == "kendall":
        tau, p_value = stats.kendalltau(x, y, alternative=alternative)
        
        xties = np.array([k for k in np.unique(x, return_counts=True)[1] if k > 1])
        yties = np.array([k for k in np.unique(y, return_counts=True)[1] if k > 1])

        T0 = n * (n - 1) / 2
        T1 = sum(xties * (xties - 1)) / 2
        T2 = sum(yties * (yties - 1)) / 2
        S = tau * np.sqrt((T0 - T1) * (T0 - T2))
        
        v0 = n * (n - 1) * (2 * n + 5)
        vt = sum(xties * (xties - 1) * (2 * xties + 5))
        vu = sum(yties * (yties - 1) * (2 * yties + 5))
        v1 = sum(xties * (xties - 1)) * sum(yties * (yties - 1))
        v2 = sum(xties * (xties - 1) * (xties - 2)) * sum(yties * (yties - 1) * (yties - 2))
        
        var_S = (v0 - vt - vu) / 18 + v1 / (2 * n * (n - 1)) + v2 / (9 * n * (n - 1) * (n - 2))
        
        if continuity:
            S = np.sign(S) * (np.abs(S) - 1)
        
        z_statistic = S / np.sqrt(var_S)
        
        names = {
            "statistic": "z",
            "estimate": "tau",
            "alternative": f"true tau is {direction} 0"
        }
        
        res.update({
            "method": "Kendall's rank correlation tau",
            "estimate": tau,
            "statistic": z_statistic,
            "p_value": p_value
        })
        
        return TestResults(res, names)

    else:
        raise ValueError("Invalid method")

def __chisq_test(x, y, p, correct, rescale_p):
        
    names = {
        "statistic": "X-squared",
        "estimate": "expected frequencies",
        "alternative": "true frequencies are not equal to expected frequencies"
    }
    
    # Convert x to a numpy array if it's a list
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, Series):
        x = x.values
    
    if not (isinstance(x, np.ndarray) and len(x.shape) == 2) and y is not None:
        if len(x) != len(y):
            raise ValueError("'x' and 'y' must have the same length")
    # If x is a matrix
    if isinstance(x, np.ndarray) and len(x.shape) == 2:
        observed = x 
    # If x and y are provided as vectors
    elif y is not None:
        observed = crosstab(np.array(x), np.array(y)).values
    # If only x is provided (goodness-of-fit test)
    elif y is None:
        observed = np.array(x).reshape(-1)
        method = "Chi-squared test for given probabilities"
        if p is not None:
            if rescale_p:
                total_p = sum(p)
                p = [prob/total_p for prob in p]
            p = np.array(p)
            if abs(p.sum() - 1) >= 1e-5:
                raise ValueError("Elements of 'p' must sum to 1")
            if np.any(p <= 0):
                raise ValueError("Elements of 'p' must be positive")
            expected = p * np.sum(observed)
            chi2_statistic = np.sum((observed - expected)**2 / expected)
            df = len(x) - 1
            if np.any(expected < 5) and np.isfinite(df):
                warnings.warn("Chi-squared approximation may be incorrect")
            p_value = 1 - stats.chi2.cdf(chi2_statistic, df)
            res = {
                "method": method,
                "statistic": chi2_statistic,
                "df": df,
                "expected": expected,
                "p": p,
                "p_value": p_value
            }
            
            return TestResults(res, names)
        else:
            p = [1/len(x)] * len(x)
            expected = np.array(p) * np.sum(observed)
            chi2_statistic = np.sum((observed - expected)**2 / expected)
            df = len(x) - 1
            if np.any(expected < 5) and np.isfinite(df):
                warnings.warn("Chi-squared approximation may be incorrect")
            p_value = 1 - stats.chi2.cdf(chi2_statistic, df)
            res = {
                "method": method,
                "statistic": chi2_statistic,
                "df": df,
                "expected": expected,
                "p": p,
                "p_value": p_value
            }
            
            return TestResults(res, names)
    else:
        raise ValueError("Invalid input. Provide either a matrix or two vectors.")

    n = np.sum(observed)
    if observed.shape == (2, 2) and correct:
        # Applying Yates' continuity correction for 2x2 table
        expected = np.outer(np.sum(observed, axis=1), np.sum(observed, axis=0)) / n
        chi2_statistic = np.sum((np.abs(observed - expected) - 0.5)**2 / expected)
        add_correct = True
    else:
        add_correct = False
        expected = np.outer(np.sum(observed, axis=1), np.sum(observed, axis=0)) / n
        chi2_statistic = np.sum((observed - expected)**2 / expected)

    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p_value = 1 - stats.chi2.cdf(chi2_statistic, df)
    
    res = {
        "method": "Pearson's Chi-squared test" + (" with Yates' continuity correction" if add_correct else ""),
        "statistic": chi2_statistic,
        "df": df,
        "expected": expected,
        "p_value": p_value
    }
    
    return TestResults(res, names)


    
