import pytest
import pandas as pd
from statsmodels.api import OLS, GLM
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from estyp.linear_model import Stepwise, LogisticRegression

# Datos de prueba
data = pd.DataFrame({
    "y": [1, 2, 3, 4, 5],
    "x1": [5, 20, 3, 2, 1],
    "x2": [6, 7, 8, 9, 10]
})


def test_stepwise_basic():
    # Test básico de inicialización y fit
    stepwise = Stepwise(formula="y ~ 1", data=data, model=OLS)
    stepwise.fit()
    assert stepwise.optimal_formula_ is not None


def test_stepwise_forward():
    # Test de selección hacia adelante
    stepwise = Stepwise(formula="y ~ 1", data=data, model=OLS, direction="forward")
    stepwise.fit()
    assert stepwise.optimal_formula_ is not None


def test_stepwise_backward():
    # Test de selección hacia atrás
    stepwise = Stepwise(formula="y ~ x1 + x2", data=data, model=OLS, direction="backward")
    stepwise.fit()
    assert stepwise.optimal_formula_ is not None


def test_stepwise_both():
    # Test de selección bidireccional
    stepwise = Stepwise(formula="y ~ 1", data=data, model=OLS, direction="both")
    stepwise.fit()
    assert stepwise.optimal_formula_ is not None


def test_stepwise_alternate_models():
    # Test utilizando diferentes modelos
    stepwise = Stepwise(formula="y ~ 1", data=data, model=GLM)
    with pytest.warns(PerfectSeparationWarning):
        stepwise.fit()
    assert stepwise.optimal_formula_ is not None

    stepwise = Stepwise(formula="y ~ 1", data=data, model=LogisticRegression)
    with pytest.raises(Exception):
        # Esperamos una excepción porque los datos no son adecuados para una regresión logística
        stepwise.fit()


def test_stepwise_invalid_parameters():
    # Test para parámetros inválidos
    with pytest.raises(ValueError):
        Stepwise(formula="y ~ 1", data=data, model=OLS, direction="invalid_direction")

    with pytest.raises(ValueError):
        Stepwise(formula="y ~ 1", data=data, model=OLS, criterion="invalid_criterion")


def test_plot_history():
    # Test del método plot_history
    stepwise = Stepwise(formula="y ~ 1", data=data, model=OLS)
    stepwise.fit()
    fig, ax = stepwise.plot_history()
    assert ax is not None
