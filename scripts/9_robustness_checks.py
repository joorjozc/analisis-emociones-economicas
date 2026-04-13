"""
Script 9: Robustness checks
Tesis doctoral — Jorge José Zamora, UPCT

Comprobaciones de robustez:
- A: Tests de cointegración (Engle-Granger, Johansen)
- B: Robustez del orden de rezagos VAR (VAR(2), VAR(4) vs VAR(6))
- C: Robustez del ordenamiento Cholesky en FEVD + GFEVD generalizada (Pesaran-Shin)
- D: Regenerar FEVD completa a 12 horizontes
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import json
import warnings
import os
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats

warnings.filterwarnings('ignore')

# Configuración
INPUT_FILE = 'econometric_dataset_global.csv'
OUTPUT_DIR = 'robustness_results'
VAR_DIR = 'var_granger_results'
IPC_COL = 'IPC_ECOICOP Índice General'
EMOTION_VARS = ['Ira', 'Miedo', 'Tristeza', 'Alegría']
ENDOGENOUS = [IPC_COL] + EMOTION_VARS

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("COMPROBACIONES DE ROBUSTEZ — TESIS DOCTORAL")
print("=" * 80)

# Cargar y preparar datos (misma lógica que script 5)
print("\n[1/5] Cargando datos...")
df = pd.read_csv(INPUT_FILE, parse_dates=['Fecha'])
df = df.set_index('Fecha').sort_index()
df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
df_var = df[ENDOGENOUS].dropna()
print(f"   ✓ {len(df_var)} obs × {len(ENDOGENOUS)} variables")

# Determinar qué variables necesitan diferenciación
needs_diff = []
for col in ENDOGENOUS:
    adf = adfuller(df_var[col], autolag='AIC')
    if adf[1] > 0.05:
        needs_diff.append(col)
print(f"   No estacionarias (necesitan diferenciación): {needs_diff}")

if needs_diff:
    df_model = df_var.diff().dropna()
else:
    df_model = df_var

print(f"   Tras diferenciación: {len(df_model)} obs")

# ═══════════════════════════════════════════════════════════════════════════════
# A. TESTS DE COINTEGRACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("[2/5] TESTS DE COINTEGRACIÓN")
print("=" * 80)

coint_results = {}

# Solo testear entre variables I(1): IPC y Alegría (típicamente)
i1_vars = [col for col in ENDOGENOUS if col in needs_diff]
print(f"   Variables I(1): {i1_vars}")

if len(i1_vars) >= 2:
    # Test de Engle-Granger
    print("\n   Test de cointegración Engle-Granger (IPC ~ Alegría)...")
    eg_stat, eg_pvalue, eg_crit = coint(df_var[IPC_COL], df_var['Alegría'], trend='c')
    print(f"   Estadístico: {eg_stat:.4f}, p-value: {eg_pvalue:.4f}")
    print(f"   Valores críticos: 1%={eg_crit[0]:.4f}, 5%={eg_crit[1]:.4f}, 10%={eg_crit[2]:.4f}")

    coint_results['engle_granger'] = {
        'statistic': round(float(eg_stat), 4),
        'p_value': round(float(eg_pvalue), 4),
        'critical_values': {
            '1pct': round(float(eg_crit[0]), 4),
            '5pct': round(float(eg_crit[1]), 4),
            '10pct': round(float(eg_crit[2]), 4),
        },
        'cointegrated_5pct': bool(eg_pvalue <= 0.05),
        'variables': [IPC_COL, 'Alegría']
    }

    # Test de Johansen (traza)
    print("\n   Test de Johansen (traza) ([IPC, Alegría])...")
    joh_data = df_var[[IPC_COL, 'Alegría']].values
    joh = coint_johansen(joh_data, det_order=0, k_ar_diff=5)
    trace_stats = joh.lr1
    trace_crit_95 = joh.cvt[:, 1]
    trace_crit_99 = joh.cvt[:, 2]

    print(f"   Test traza r=0: stat={trace_stats[0]:.4f}, cv 95%={trace_crit_95[0]:.4f}")
    print(f"   Test traza r=1: stat={trace_stats[1]:.4f}, cv 95%={trace_crit_95[1]:.4f}")

    coint_results['johansen'] = {
        'trace_statistics': [round(float(s), 4) for s in trace_stats],
        'critical_values_95': [round(float(c), 4) for c in trace_crit_95],
        'critical_values_99': [round(float(c), 4) for c in trace_crit_99],
        'reject_r0_at_5pct': bool(trace_stats[0] > trace_crit_95[0]),
        'reject_r1_at_5pct': bool(trace_stats[1] > trace_crit_95[1]),
        'variables': [IPC_COL, 'Alegría'],
        'det_order': 0,
        'k_ar_diff': 5
    }

    cointegrated = coint_results['engle_granger']['cointegrated_5pct'] or coint_results['johansen']['reject_r0_at_5pct']
    coint_results['cointegration_found'] = cointegrated
    print(f"\n   ¿Cointegración detectada al 5%? {cointegrated}")

    if cointegrated:
        print("   Estimando VECM como comprobación de robustez...")
        from statsmodels.tsa.vector_ar.vecm import VECM
        vecm_data = df_var[ENDOGENOUS]
        vecm = VECM(vecm_data, k_ar_diff=5, coint_rank=1, deterministic='ci')
        vecm_fit = vecm.fit()
        coint_results['vecm_note'] = 'VECM(5) con rango=1 estimado como comprobación de robustez'
        print("   VECM estimado correctamente")
    else:
        coint_results['vecm_note'] = 'No se encontró cointegración; la especificación VAR diferenciado es apropiada'
else:
    coint_results['note'] = f'Solo {len(i1_vars)} variable(s) I(1); test de cointegración no aplicable'
    coint_results['cointegration_found'] = False

with open(os.path.join(OUTPUT_DIR, 'cointegration_tests.json'), 'w', encoding='utf-8') as f:
    json.dump(coint_results, f, ensure_ascii=False, indent=2)
print(f"   ✓ cointegration_tests.json")

# ═══════════════════════════════════════════════════════════════════════════════
# B. ROBUSTEZ DEL ORDEN DE REZAGOS VAR
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("[3/5] ROBUSTEZ DEL ORDEN DE REZAGOS VAR")
print("=" * 80)

var_robustness = {}

for lag_order in [2, 4, 6]:
    print(f"\n   --- VAR({lag_order}) ---")
    try:
        model = VAR(df_model)
        var_fit = model.fit(lag_order)
        n_obs = var_fit.nobs
        n_vars = len(ENDOGENOUS)
        n_params = n_vars * lag_order * n_vars + n_vars
        param_obs_ratio = n_params / n_obs

        # Test de Portmanteau
        port_test = var_fit.test_whiteness(nlags=lag_order + 5)
        port_pvalue = float(port_test.pvalue)

        # Causalidad de Granger (pares significativos)
        sig_granger = []
        for cause in ENDOGENOUS:
            for effect in ENDOGENOUS:
                if cause == effect:
                    continue
                try:
                    test_data = df_model[[effect, cause]].dropna()
                    gc = grangercausalitytests(test_data, maxlag=lag_order, verbose=False)
                    for lag in range(1, lag_order + 1):
                        if lag in gc:
                            f_test = gc[lag][0]['ssr_ftest']
                            if f_test[1] <= 0.05:
                                sig_granger.append({
                                    'cause': cause, 'effect': effect,
                                    'lag': lag, 'f_stat': round(f_test[0], 4),
                                    'p_value': round(f_test[1], 4)
                                })
                except:
                    pass

        # Contar pares direccionales únicos
        unique_pairs = set()
        for sg in sig_granger:
            unique_pairs.add((sg['cause'], sg['effect']))

        # FEVD en horizonte 12
        fevd = var_fit.fevd(periods=12)
        fevd_decomp = fevd.decomp
        ipc_idx = ENDOGENOUS.index(IPC_COL)

        # Detectar forma del array: (n_vars, periods, n_vars) vs (periods, n_vars, n_vars)
        shape = fevd_decomp.shape
        if shape[0] == n_vars and shape[1] > shape[0]:
            n_periods_avail = shape[1]
            last_h = n_periods_avail - 1
            fevd_h12 = {ENDOGENOUS[j]: round(float(fevd_decomp[ipc_idx, last_h, j]), 4)
                        for j in range(n_vars)}
        else:
            n_periods_avail = shape[0]
            last_h = n_periods_avail - 1
            fevd_h12 = {ENDOGENOUS[j]: round(float(fevd_decomp[last_h, ipc_idx, j]), 4)
                        for j in range(n_vars)}

        # Resultados clave de Granger
        emo_to_cpi = [p for p in unique_pairs if p[1] == IPC_COL and p[0] != IPC_COL]
        cpi_to_emo = [p for p in unique_pairs if p[0] == IPC_COL and p[1] != IPC_COL]

        var_robustness[f'VAR({lag_order})'] = {
            'lag_order': lag_order,
            'n_obs': int(n_obs),
            'n_params_total': n_params,
            'param_obs_ratio': round(param_obs_ratio, 2),
            'portmanteau_pvalue': round(port_pvalue, 4),
            'n_significant_granger_pairs': len(unique_pairs),
            'emotions_cause_cpi': [p[0] for p in emo_to_cpi],
            'cpi_causes_emotions': [p[1] for p in cpi_to_emo],
            'fevd_cpi_horizon12': fevd_h12,
            'significant_granger_details': sig_granger
        }

        print(f"   Obs: {n_obs}, Params: {n_params}, Ratio: {param_obs_ratio:.2f}")
        print(f"   Portmanteau p: {port_pvalue:.4f}")
        print(f"   Pares Granger significativos: {len(unique_pairs)}")
        print(f"   Emociones→IPC: {[p[0] for p in emo_to_cpi]}")
        print(f"   IPC→Emociones: {[p[1] for p in cpi_to_emo]}")
        print(f"   FEVD h12: {fevd_h12}")

    except Exception as e:
        print(f"   Error: {e}")
        var_robustness[f'VAR({lag_order})'] = {'error': str(e)}

with open(os.path.join(OUTPUT_DIR, 'var_robustness.json'), 'w', encoding='utf-8') as f:
    json.dump(var_robustness, f, ensure_ascii=False, indent=2)
print(f"\n   ✓ var_robustness.json")

# ═══════════════════════════════════════════════════════════════════════════════
# C. ROBUSTEZ DEL ORDENAMIENTO CHOLESKY EN FEVD
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("[4/5] ROBUSTEZ DEL ORDENAMIENTO FEVD")
print("=" * 80)

# Definir ordenamientos alternativos
orderings = {
    'baseline': [IPC_COL, 'Ira', 'Miedo', 'Tristeza', 'Alegría'],
    'joy_first': ['Alegría', IPC_COL, 'Ira', 'Miedo', 'Tristeza'],
    'joy_second': [IPC_COL, 'Alegría', 'Ira', 'Miedo', 'Tristeza'],
    'reversed': ['Alegría', 'Tristeza', 'Miedo', 'Ira', IPC_COL],
    'emotions_first': ['Ira', 'Miedo', 'Tristeza', 'Alegría', IPC_COL],
}

fevd_ordering_results = {}
fevd_table_rows = []

for order_name, ordering in orderings.items():
    print(f"\n   --- Ordenamiento: {order_name} ---")
    print(f"   {ordering}")

    df_reordered = df_model[ordering]
    model = VAR(df_reordered)
    var_fit = model.fit(6)
    fevd = var_fit.fevd(periods=12)
    fevd_decomp = fevd.decomp

    # Detectar forma del array
    shape = fevd_decomp.shape
    if shape[0] == len(ordering) and shape[1] > shape[0]:
        n_periods_avail = shape[1]
    else:
        n_periods_avail = shape[0]

    cpi_idx = ordering.index(IPC_COL)

    fevd_cpi = {}
    for h in [1, 3, 6, 12]:
        h_idx = min(h - 1, n_periods_avail - 1)
        if shape[0] == len(ordering) and shape[1] > shape[0]:
            fevd_at_h = {ordering[j]: round(float(fevd_decomp[cpi_idx, h_idx, j]), 4)
                         for j in range(len(ordering))}
        else:
            fevd_at_h = {ordering[j]: round(float(fevd_decomp[h_idx, cpi_idx, j]), 4)
                         for j in range(len(ordering))}
        fevd_cpi[f'h{h}'] = fevd_at_h

    fevd_ordering_results[order_name] = {
        'ordering': ordering,
        'fevd_cpi': fevd_cpi
    }

    # Fila para tabla h=12
    h12 = fevd_cpi['h12']
    fevd_table_rows.append({
        'Ordering': order_name,
        'CPI': h12.get(IPC_COL, 0),
        'Ira': h12.get('Ira', 0),
        'Miedo': h12.get('Miedo', 0),
        'Tristeza': h12.get('Tristeza', 0),
        'Alegría': h12.get('Alegría', 0),
    })

    print(f"   FEVD IPC h12: IPC={h12.get(IPC_COL, 0):.4f}, "
          f"Ira={h12.get('Ira', 0):.4f}, Miedo={h12.get('Miedo', 0):.4f}, "
          f"Tristeza={h12.get('Tristeza', 0):.4f}, Alegría={h12.get('Alegría', 0):.4f}")

# GFEVD generalizada (Pesaran-Shin, invariante al orden)
print("\n   --- GFEVD Generalizada (Pesaran-Shin) ---")

model_base = VAR(df_model[ENDOGENOUS])
var_base = model_base.fit(6)
n_vars = len(ENDOGENOUS)
periods = 12

# Coeficientes MA (matrices Phi)
ma_coefs = np.array(var_base.ma_rep(periods - 1))  # shape: (periods, n_vars, n_vars)

# Matriz de covarianza residual
sigma = np.array(var_base.sigma_u)  # (n_vars, n_vars)
sigma_diag = np.diag(sigma)  # elementos diagonales (varianzas)

# Cálculo GFEVD
# theta_ij(h) = sum_{l=0}^{h} (e_i' Phi_l Sigma e_j)^2 / sigma_jj
# dividido por FEV total: sum_{l=0}^{h} e_i' Phi_l Sigma Phi_l' e_i

gfevd = np.zeros((periods, n_vars, n_vars))

for h in range(periods):
    for i in range(n_vars):
        # Varianza total del error de predicción para variable i en horizonte h
        total_fev = 0
        for l in range(h + 1):
            phi_l = ma_coefs[l]
            total_fev += phi_l[i, :] @ sigma @ phi_l[i, :]

        for j in range(n_vars):
            # Numerador: contribución generalizada del shock j a variable i
            numerator = 0
            for l in range(h + 1):
                phi_l = ma_coefs[l]
                numerator += (phi_l[i, :] @ sigma[:, j]) ** 2
            numerator /= sigma_diag[j]

            gfevd[h, i, j] = numerator / total_fev if total_fev > 0 else 0

    # Normalizar filas para que sumen 1
    for i in range(n_vars):
        row_sum = gfevd[h, i, :].sum()
        if row_sum > 0:
            gfevd[h, i, :] /= row_sum

# Extraer GFEVD para IPC
ipc_idx = ENDOGENOUS.index(IPC_COL)
gfevd_cpi = {}
for h in [1, 3, 6, 12]:
    gfevd_cpi[f'h{h}'] = {ENDOGENOUS[j]: round(float(gfevd[h-1, ipc_idx, j]), 4)
                           for j in range(n_vars)}

fevd_ordering_results['generalized'] = {
    'ordering': 'invariante al orden (Pesaran-Shin)',
    'fevd_cpi': gfevd_cpi
}

h12_gen = gfevd_cpi['h12']
fevd_table_rows.append({
    'Ordering': 'Generalizada (P-S)',
    'CPI': h12_gen.get(IPC_COL, 0),
    'Ira': h12_gen.get('Ira', 0),
    'Miedo': h12_gen.get('Miedo', 0),
    'Tristeza': h12_gen.get('Tristeza', 0),
    'Alegría': h12_gen.get('Alegría', 0),
})

print(f"   GFEVD IPC h12: IPC={h12_gen.get(IPC_COL, 0):.4f}, "
      f"Ira={h12_gen.get('Ira', 0):.4f}, Miedo={h12_gen.get('Miedo', 0):.4f}, "
      f"Tristeza={h12_gen.get('Tristeza', 0):.4f}, Alegría={h12_gen.get('Alegría', 0):.4f}")

# Guardar resultados FEVD ordering
with open(os.path.join(OUTPUT_DIR, 'fevd_ordering_robustness.json'), 'w', encoding='utf-8') as f:
    json.dump(fevd_ordering_results, f, ensure_ascii=False, indent=2)

df_fevd_table = pd.DataFrame(fevd_table_rows)
df_fevd_table.to_csv(os.path.join(OUTPUT_DIR, 'fevd_ordering_table.csv'), index=False, encoding='utf-8-sig')
print(f"\n   ✓ fevd_ordering_robustness.json, fevd_ordering_table.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# D. REGENERAR FEVD COMPLETA A 12 HORIZONTES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("[5/5] REGENERANDO FEVD COMPLETA A 12 HORIZONTES")
print("=" * 80)

model_full = VAR(df_model[ENDOGENOUS])
var_full = model_full.fit(6)
fevd_full = var_full.fevd(periods=12)
fevd_decomp_full = fevd_full.decomp
print(f"   FEVD decomp shape: {fevd_decomp_full.shape}")

ipc_idx = ENDOGENOUS.index(IPC_COL)

# Determinar layout de ejes correcto
shape = fevd_decomp_full.shape
if shape[0] == len(ENDOGENOUS) and shape[1] > shape[0]:
    n_horizons_full = shape[1]
    fevd_ipc_data = fevd_decomp_full[ipc_idx, :, :]
else:
    n_horizons_full = shape[0]
    fevd_ipc_data = fevd_decomp_full[:, ipc_idx, :]

print(f"   Horizontes: {n_horizons_full}")
fevd_ipc_full = pd.DataFrame(
    fevd_ipc_data,
    columns=ENDOGENOUS,
    index=range(1, n_horizons_full + 1)
)
fevd_ipc_full.index.name = 'Horizonte (meses)'

# Guardar en directorio de robustez
fevd_ipc_full.to_csv(os.path.join(OUTPUT_DIR, 'fevd_ipc_12h.csv'), encoding='utf-8-sig')

# Actualizar archivo original con datos correctos
fevd_ipc_full.to_csv(os.path.join(VAR_DIR, 'fevd_ipc.csv'), encoding='utf-8-sig')

# Actualizar summary JSON
summary_file = os.path.join(VAR_DIR, 'var_granger_summary.json')
if os.path.exists(summary_file):
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    summary['fevd_ipc_horizon12'] = {k: round(v, 4) for k, v in fevd_ipc_full.iloc[-1].to_dict().items()}
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print(f"   FEVD completa a 12 horizontes:")
print(fevd_ipc_full.round(4).to_string())
print(f"\n   ✓ fevd_ipc_12h.csv (y actualizado var_granger/fevd_ipc.csv)")

# ═══════════════════════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("✓ COMPROBACIONES DE ROBUSTEZ COMPLETAS")
print("=" * 80)
print(f"\nOutputs guardados en: {OUTPUT_DIR}")
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    print(f"   {f_name}")
