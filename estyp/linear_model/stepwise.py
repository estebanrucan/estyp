import statsmodels.api as sm
import warnings
from pandas import DataFrame
warnings.filterwarnings("ignore")

def both_selection(formula: str, data: DataFrame, model: sm.GLM, max_iter = 10000) -> str:
    """
# Both Forward and Backward Variable Selection for GLM's

This function performs both forward and backward variable selection using the Akaike Information Criterion (AIC).

## Parameters

* `formula`: A string representing the initial model formula.
* `data`: A Pandas DataFrame containing the data to be used for model fitting.
* `model`: A statsmodels.GLM object that represents the type of model to be fit.
* `max_iter`: The maximum number of iterations to perform.

## Returns

A string representing the final model formula.

## Example

```python
import statsmodels.api as sm
import pandas as pd

data = pd.DataFrame({
    "y": [1, 2, 3, 4, 5],
    "x1": [1, 2, 3, 4, 5],
    "x2": [6, 7, 8, 9, 10],
})

formula = "y ~ x1 + x2"

model = sm.GLM

final_formula = both_selection(formula=formula, data=data, model=model)

print(final_formula)
    """
    print("Este proceso tarda un buen tiempo ¡Paciencia! Momento de cuestionarte si eres feliz.")
    # Preparación
    fi = formula
    e = model.from_formula(fi, data)
    m = e.fit()
    preds = e.formula.split("+")
    vr = preds[0].split("~")[0].strip()
    preds[0] = preds[0].split("~")[1]
    preds = [x.strip() for x in preds]
    
    # Algoritmo
    d = data.copy()

    aicf = m.aic
    aic0 = 0

    var_el = []
    var_ag = []
    print(f"AIC Inicial: {aicf:0.2f}")

    for i in range(max_iter):
        if i == 0:
            aics = []
            for p in preds:
                # Crear formula
                p0 = d.drop(columns=[p, vr]).columns.tolist()
                f0 = f"{vr} ~ {' + '.join(p0)}"
                # Ajustar modelo y extraer AIC
                m0 = model.from_formula(f0, d).fit()
                aics.append(m0.aic)
            aic0 = min(aics)
            id_min_aic = aics.index(min(aics))
            if aic0 < aicf:
                var_min = preds[id_min_aic]
                data0 = d.drop(columns=var_min)
                preds0 = data0.drop(columns=vr).columns.tolist()
                ff = f"{vr} ~ {' + '.join(preds0)}"
                var_el.append(var_min)
            else:
                print("No hay mejoras en el AIC, por lo que no se eliminarán variables.")
                return fi
        else:
            aicf = aic0
            # Quitar variable
            aics_a = []
            for p in preds0:
                p0 = data0.drop(columns=[p, vr]).columns.tolist()
                f0 = f"{vr} ~ {' + '.join(p0)}"
                m0 = model.from_formula(f0, d).fit()
                aics_a.append(m0.aic)
            # Añadir variable
            aics_b = []
            for p in preds:
                f0 = f"{ff} + {p}"
                m0 = model.from_formula(f0, d).fit()
                aics_b.append(m0.aic)
            min_a, min_b = min(aics_a), min(aics_b)
            aic0 = min_a if min_a < min_b else min_b
            if aic0 < aicf:
                if min_a < min_b:
                    var_min = preds0[aics_a.index(min_a)]
                    data0.drop(columns=var_min, inplace=True)
                    preds0 = data0.drop(columns=vr).columns.tolist()
                    ff = f"{vr} ~ {' + '.join(preds0)}"
                    var_el.append(var_min)
                else:
                    var_min = preds[aics_b.index(min_b)]
                    data0 = d[preds0 + [var_min, vr]]
                    preds0.append(var_min)
                    ff = f"{vr} ~ {' + '.join(preds0)}"
                    var_ag.append(var_min)
                aicf = aic0
            else:
                print("=========================")
                print("|| Fin de la selección ||")
                print("=========================")
                print(f"AIC Obtenido: {aicf:0.2f}")
                print("Variables eliminadas:", var_el if var_el else "Ninguna")
                print("Variables agregadas:", var_ag if var_ag else "Ninguna")
                print("Formula obtenida:", ff)
                return ff
        print(f"Iteración {i + 1} Finalizada | AIC Actual: {aic0:0.2f}")
        if not preds0:
            return f"{vr} + 1"
    print("Máximas iteraciones alcanzadas")
    return ff