import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from estyp.linear_model import LogisticRegression

@pytest.fixture
def sample_data():
    np.random.seed(123)
    data = pd.DataFrame({
        "y": np.random.randint(2, size=100),
        "x1": np.random.uniform(-1, 1, size=100),
        "x2": np.random.uniform(-1, 1, size=100),
    })
    return data

def test_initialization(sample_data):
    X = sample_data[['x1', 'x2']]
    y = sample_data['y']
    model = LogisticRegression(X, y)
    assert model.X.shape == (100, 3)
    assert model.y.shape == (100,)

def test_from_formula(sample_data):
    formula = "y ~ x1 + x2"
    model = LogisticRegression.from_formula(formula, sample_data)
    assert model.X.shape == (100, 3)
    assert model.y.shape == (100,)

def test_fit(sample_data):
    formula = "y ~ x1 + x2"
    model = LogisticRegression.from_formula(formula, sample_data)
    fitted_model = model.fit()
    assert_almost_equal(fitted_model.coef_.flatten(), [-0.2, 0.03, 0.44], decimal=2) # Check with expected coefficients

def test_predict(sample_data):
    formula = "y ~ x1 + x2"
    model = LogisticRegression.from_formula(formula, sample_data)
    model.fit()
    predictions = model.predict(sample_data)
    assert len(predictions) == 100

def test_properties(sample_data):
    formula = "y ~ x1 + x2"
    model = LogisticRegression.from_formula(formula, sample_data)
    model.fit()
    assert_almost_equal(model.aic, 135.96, decimal=2)
    assert_almost_equal(model.bic, 135.96, decimal=2)
    assert model.cov_matrix.shape == (3, 3)
    assert model.params.shape == (3,)

def test_summary_methods(sample_data):
    formula = "y ~ x1 + x2"
    model = LogisticRegression.from_formula(formula, sample_data)
    model.fit()
    assert model.conf_int().shape == (3, 2)
    assert model.se().shape == (3,)
    assert model.z_values().shape == (3,)
    assert model.p_values().shape == (3,)
    assert model.summary().shape == (3, 6)
