from scipy.stats import f as fisher
from estyp.testing.__base import __nested_models_test




def __both_selection(formula, data, model, max_iter) -> str:
    
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
                f0 = f"{vr} ~ {' + '.join(p0) if p0 else 1}"
                # Ajustar modelo y extraer AIC
                m0 = model.from_formula(f0, d).fit()
                aics.append(m0.aic)
            aic0 = min(aics)
            id_min_aic = aics.index(min(aics))
            if aic0 < aicf:
                var_min = preds[id_min_aic]
                data0 = d.drop(columns=var_min)
                preds0 = data0.drop(columns=vr).columns.tolist()
                ff = f"{vr} ~ {' + '.join(preds0) if preds0 else 1}"
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
                f0 = f"{vr} ~ {' + '.join(p0) if p0 else 1}"
                m0 = model.from_formula(f0, d).fit()
                aics_a.append(m0.aic)
            # Añadir variable
            aics_b = []
            for p in preds:
                f0 = f"{ff} {('+ ' + p) if p not in preds0 else ''}"
                m0 = model.from_formula(f0, d).fit() if p not in preds0 else None
                aics_b.append(m0.aic if p not in preds0 else aics_a[preds0.index(p)])
            min_a, min_b = min(aics_a), min(aics_b)
            aic0 = min_a if min_a < min_b else min_b
            if aic0 < aicf:
                if min_a < min_b:
                    var_min = preds0[aics_a.index(min_a)]
                    data0.drop(columns=var_min, inplace=True)
                    preds0 = data0.drop(columns=vr).columns.tolist()
                    ff = f"{vr} ~ {' + '.join(preds0) if preds0 else 1}"
                    var_el.append(var_min)
                else:
                    var_min = preds[aics_b.index(min_b)]
                    data0 = d[preds0 + [var_min, vr]]
                    preds0.append(var_min)
                    ff = f"{vr} ~ {' + '.join(preds0) if preds0 else 1}"
                    var_ag.append(var_min)
                aicf = aic0
            else:
                print("=========================")
                print("|| Fin de la selección ||")
                print("=========================")
                print(f"AIC Obtenido: {aicf:0.2f}")
                print("Variables eliminadas:", var_el if var_el else "Ninguna")
                print("Variables agregadas:", var_ag if var_ag else "Ninguna")
                print("Fórmula obtenida:", ff)
                return ff
        print(f"Iteración {i + 1} Finalizada | AIC Actual: {aic0:0.2f}")
        if not preds0:
            return f"{vr} + 1"
    print("Máximas iteraciones alcanzadas")
    return ff

def __forward_selection(y, data, model, alpha):
    preds = data.columns.to_list()
    preds.remove(y)

    f_actual = f"{y} ~ 1"
    m_actual = model.from_formula(f_actual, data).fit()
    termino = False
    
    while not termino:
        valores_p = []
        for p in preds:
            f_prueba = f"{f_actual} + {p}"
            m_prueba = model.from_formula(f_prueba, data).fit()
            pv = __nested_models_test(m_actual, m_prueba, data[y]).p_value
            valores_p.append(pv)
        min_vp = min(valores_p)
        if min_vp >= alpha:
            termino = True
            f_actual = f_actual.replace(" 1 +", "")
            m_actual = model.from_formula(f_actual, data).fit()
        else:
            var_min = preds[valores_p.index(min_vp)]
            f_actual = f"{f_actual} + {var_min}"
            m_actual = model.from_formula(f_actual, data).fit()
            preds.remove(var_min)
            termino = False if preds else True
            vp = f"{min_vp:0.4f}" if min_vp >= 0.0001 else "<0.0001"
            print(f"Variable agregada: {var_min:30} | valor-p: {vp}")
    print("=========================")
    print("|| Fin de la selección ||")
    print("=========================")
    print("Fórmula obtenida:", f_actual)
    return f_actual