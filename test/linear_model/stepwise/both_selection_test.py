import statsmodels.api as sm
from estyp.linear_model import LogisticRegression
import pandas as pd
from estyp.linear_model.stepwise import both_selection
import pytest
from sklearn.datasets import load_iris

@pytest.fixture
def data():
    content = load_iris()
    X = pd.DataFrame(content.data, columns=[f"x{i+1}" for i in range(4)])
    y = (pd.Series(content.target, name="y") == 1).astype(int)
    data = pd.concat([X, y], axis=1)
    return data

def test_both_selection_ols(data):
    formula = "x1 ~ x2 + x3 + x4 + y"
    model = sm.OLS
    with pytest.deprecated_call():
        final_formula = both_selection(formula=formula, data=data, model=model)
    assert final_formula == "x1 ~ x2 + x3 + x4", "Formula doesn't match expected"

def test_both_selection_logit(data):
    formula = "y ~ x1 + x2 + x3 + x4"
    model = sm.Logit
    with pytest.deprecated_call():
        final_formula = both_selection(formula=formula, data=data, model=model)
    assert final_formula == "y ~ x2 + x3 + x4", "Formula doesn't match expected"
    
def test_both_selection_logisticregression(data):
    formula = "y ~ x1 + x2 + x3 + x4"
    model = LogisticRegression
    with pytest.deprecated_call():
        final_formula = both_selection(formula=formula, data=data, model=model)
    assert final_formula == "y ~ x1 + x2 + x3 + x4", "Formula doesn't match expected"