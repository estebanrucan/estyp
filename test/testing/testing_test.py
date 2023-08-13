import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
import pytest

from estyp.testing import (
    CheckModel,
    chisq_test,
    cor_test,
    nested_models_test,
    prop_test,
    t_test,
    var_test,
)

@pytest.mark.filterwarnings("ignore")
def test_check_all():
    diabetes = load_diabetes()
    X = diabetes["data"]
    y = diabetes["target"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    fitted_model = model.fit()
    cm = CheckModel(fitted_model)

    # Prueba con diferentes combinaciones de parámetros
    assert isinstance(cm.check_all(alpha=0.05, plot=False, return_vals=True), dict)
    assert cm.check_all(alpha=0.01, plot=False, return_vals=False) is None


def test_var_test():
    # Caso con varianzas iguales
    np.random.seed(123)
    x, y = np.random.normal(size=(2, 100), scale=1)
    result = var_test(x, y)
    assert result.p_value > 0.05

    # Caso con varianzas diferentes
    y = np.random.normal(size=100, scale=2)
    result = var_test(x, y)
    assert result.p_value < 0.05


def test_t_test():
    # Caso con media diferente de 5
    np.random.seed(123)
    x = np.random.normal(loc=2, size=100)
    result = t_test(x, mu=5)
    assert result.p_value < 0.05

    # Caso con media igual a la media de la muestra
    result = t_test(x, mu=np.mean(x))
    assert result.p_value > 0.99


def test_nested_models_test():
    np.random.seed(123)
    data = pd.DataFrame(
        {
            "x": np.random.uniform(size=100, low=-10, high=10),
        }
    )
    data["y"] = 1 + 0.7 * data["x"] + np.random.normal(size=100, scale=10)
    model_small = sm.OLS.from_formula("y ~ 1", data).fit()
    model_big = sm.OLS.from_formula("y ~ x", data).fit()
    result = nested_models_test(model_small, model_big)
    assert result.p_value < 0.05


def test_prop_test():
    # Caso con proporciones iguales
    x = [50, 25]
    n = [100, 50]
    result = prop_test(x, n)
    assert result.p_value > 0.99

    # Caso con proporciones diferentes
    x = [50, 10]
    result = prop_test(x, n)
    assert result.p_value < 0.05

    # Otro caso de proporciones diferentes
    x = [90, 10]
    result = prop_test(x)
    assert result.p_value < 0.05


def test_cor_test():
    # Caso con casi perfecta correlación
    np.random.seed(123)
    x = np.random.normal(size=100)
    y = 0.5 * x + np.random.normal(size=100, scale=0.01)
    result = cor_test(x, y)
    assert result.p_value < 0.05

    # Caso con correlación alta
    np.random.seed(123)
    y = y + np.random.normal(size=100, scale=0.5)
    result = cor_test(x, y, method="spearman")
    assert result.p_value < 0.05

    # Caso con correlación cero
    np.random.seed(123)
    y = np.random.uniform(size=100, low=-10, high=10)
    result = cor_test(x, y, method="kendall")
    assert result.p_value > 0.05


def test_chisq_test():
    # Caso con distribución uniforme
    M = [[25, 25], [25, 25]]
    result = chisq_test(M)
    assert result.p_value > 0.05

    # Caso con distribución no uniforme
    M = [[43, 8], [14, 50]]
    result = chisq_test(M)
    assert result.p_value < 0.05
