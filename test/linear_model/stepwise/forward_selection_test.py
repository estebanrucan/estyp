import statsmodels.api as sm
from estyp.linear_model import LogisticRegression
import pandas as pd
from estyp.linear_model.stepwise import forward_selection
import pytest
from sklearn.datasets import load_iris

@pytest.fixture
def data():
    content = load_iris()
    X = pd.DataFrame(content.data, columns=[f"x{i+1}" for i in range(4)])
    y = (pd.Series(content.target, name="y") == 1).astype(int)
    data = pd.concat([X, y], axis=1)
    return data

def test_forward_selection_ols(data):
    model = sm.OLS
    final_formula = forward_selection(y="x1", data=data, model=model)
    assert final_formula == "x1 ~ x3 + x2 + x4", "Formula doesn't match expected"

def test_forward_selection_logit(data):
    model = sm.Logit
    final_formula = forward_selection(y="y", data=data, model=model)
    assert final_formula == "y ~ x2", "Formula doesn't match expected"
    
def test_forward_selection_logisticregression(data):
    model = LogisticRegression
    final_formula = forward_selection(y="y", data=data, model=model)
    assert final_formula == "y ~ x2", "Formula doesn't match expected"