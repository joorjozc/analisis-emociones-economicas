"""
Script 5: Modelo VAR y causalidad de Granger
Tesis doctoral — Jorge José Zamora, UPCT

- Modelo VAR endógeno: [IPC, Ira, Miedo, Tristeza, Alegría]
- Selección de rezagos por AIC/BIC
- Tests de Granger bidireccionales
- Funciones Impulso-Respuesta (IRF)
- Descomposición de Varianza del Error de Predicción (FEVD)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import json
import warnings
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# ─── Configuración ───────────────────────────────────────────────────────────
INPUT_FILE = 'econometric_dataset_global.csv'
OUTPUT_DIR = 'var_granger_results'
IPC_COL = 'IPC_ECOICOP Índice General'
EMOTION_VARS = ['Ira', 'Miedo', 'Tristeza', 'Alegría']
ENDOGENOUS = [IPC_COL] + EMOTION_VARS
MAX_LAGS = 8  # Ajustado para consistencia metodológica con el artículo

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("MODELO VAR Y CAUSALIDAD DE GRANGER")
print("=" * 80)

# ─── Paso 1: Cargar y preparar datos ─────────────────────────────────────────
print("\n[1/6] Cargando datos...")
df = pd.read_csv(INPUT_FILE, parse_dates=['Fecha'])
df = df.set_index('Fecha').sort_index()
df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()

# Seleccionar variables endógenas
df_var = df[ENDOGENOUS].dropna()
print(f"   ✓ {len(df_var)} obs × {len(ENDOGENOUS)} variables")
print(f"   Período: {df_var.index[0]} — {df_var.index[-1]}")

# ─── Paso 2: Tests de estacionariedad ────────────────────────────────────────
print("\n[2/6] Tests ADF de estacionariedad...")
adf_results = []
needs_diff = []

for col in ENDOGENOUS:
    adf = adfuller(df_var[col], autolag='AIC')
    stationary = adf[1] <= 0.05
    adf_results.append({
        'variable': col,
        'adf_stat': round(adf[0], 4),
        'p_value': round(adf[1], 4),
        'stationary': stationary
    })
    if not stationary:
        needs_diff.append(col)
    status = "✓ Estacionaria" if stationary else "✗ No estacionaria"
    print(f"   {col[:40]:40s} ADF={adf[0]:.4f}, p={adf[1]:.4f} → {status}")

# Diferenciar si necesario
if needs_diff:
    print(f"\n   Diferenciando: {needs_diff}")
    df_diff = df_var.diff().dropna()
    # Verificar estacionariedad post-diferenciación
    for col in needs_diff:
        adf2 = adfuller(df_diff[col], autolag='AIC')
        print(f"   {col[:40]:40s} (d=1) ADF={adf2[0]:.4f}, p={adf2[1]:.4f}")
    df_model = df_diff
else:
    df_model = df_var

# ─── Paso 3: Selección de rezagos ────────────────────────────────────────────
print("\n[3/6] Selección de rezagos óptimos...")
model_selector = VAR(df_model)
lag_selection = model_selector.select_order(maxlags=MAX_LAGS)
print(lag_selection.summary())

optimal_lag_aic = lag_selection.aic
optimal_lag_bic = lag_selection.bic
optimal_lag = max(1, optimal_lag_aic)  # usar AIC, mínimo 1
print(f"\n   Rezago óptimo (AIC): {optimal_lag_aic}")
print(f"   Rezago óptimo (BIC): {optimal_lag_bic}")
print(f"   → Usando: {optimal_lag} rezagos")

# ─── Paso 4: Estimar VAR ─────────────────────────────────────────────────────
print(f"\n[4/6] Estimando VAR({optimal_lag})...")
var_model = model_selector.fit(optimal_lag)
print(var_model.summary())

# Diagnósticos
durbin_watson = var_model.test_whiteness(nlags=optimal_lag + 5).pvalue
print(f"\n   Test de Portmanteau (ruido blanco) p-value: {durbin_watson:.4f}")

# ─── Paso 5: Tests de Granger bidireccionales ────────────────────────────────
print("\n[5/6] Tests de causalidad de Granger bidireccionales...")
granger_results = []

for cause in ENDOGENOUS:
    for effect in ENDOGENOUS:
        if cause == effect:
            continue

        try:
            test_data = df_model[[effect, cause]].dropna()
            gc = grangercausalitytests(test_data, maxlag=optimal_lag, verbose=False)

            # Extraer p-values de cada rezago
            for lag in range(1, optimal_lag + 1):
                if lag in gc:
                    f_test = gc[lag][0]['ssr_ftest']
                    chi2_test = gc[lag][0]['ssr_chi2test']

                    granger_results.append({
                        'cause': cause,
                        'effect': effect,
                        'lag': lag,
                        'f_stat': round(f_test[0], 4),
                        'f_pvalue': round(f_test[1], 4),
                        'chi2_stat': round(chi2_test[0], 4),
                        'chi2_pvalue': round(chi2_test[1], 4),
                        'significant_5pct': f_test[1] <= 0.05
                    })
        except Exception as e:
            print(f"   ⚠ Error {cause} → {effect}: {e}")

df_granger = pd.DataFrame(granger_results)
print(f"\n   Tests realizados: {len(granger_results)}")

# Resumen de causalidades significativas
sig_granger = df_granger[df_granger['significant_5pct']]
print(f"\n   Causalidades significativas (p < 0.05):")
for _, row in sig_granger.iterrows():
    print(f"      {row['cause'][:30]:30s} → {row['effect'][:30]:30s} "
          f"(lag={row['lag']}, F={row['f_stat']:.3f}, p={row['f_pvalue']:.4f})")

# ─── Paso 6: IRF y FEVD ──────────────────────────────────────────────────────
print("\n[6/6] Funciones Impulso-Respuesta y Descomposición de Varianza...")

# IRF (12 períodos)
irf = var_model.irf(periods=12)

# Gráfico IRF
fig_irf = irf.plot(orth=True, figsize=(16, 12))
fig_irf.suptitle('Funciones Impulso-Respuesta Ortogonalizadas (12 meses)', fontweight='bold')
plt.tight_layout()
irf_path = os.path.join(OUTPUT_DIR, 'irf_orthogonal.png')
plt.savefig(irf_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ IRF guardado: {irf_path}")

# FEVD (12 períodos)
fevd = var_model.fevd(periods=12)

# Gráfico FEVD
fig_fevd = fevd.plot(figsize=(16, 12))
plt.suptitle('Descomposición de Varianza del Error de Predicción (12 meses)', fontweight='bold')
plt.tight_layout()
fevd_path = os.path.join(OUTPUT_DIR, 'fevd.png')
plt.savefig(fevd_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ FEVD guardado: {fevd_path}")

# Extraer tabla FEVD para IPC
fevd_summary = fevd.decomp
ipc_idx = ENDOGENOUS.index(IPC_COL)
n_vars = len(ENDOGENOUS)

# Detectar forma del array: (n_vars, periods, n_vars) vs (periods, n_vars, n_vars)
shape = fevd_summary.shape
if shape[0] == n_vars and shape[1] > shape[0]:
    # Shape: (n_vars, periods, n_vars) = [response, horizon, shock]
    n_periods_fevd = shape[1]
    fevd_ipc_data = fevd_summary[ipc_idx, :, :]
else:
    # Shape: (periods, n_vars, n_vars) = [horizon, response, shock]
    n_periods_fevd = shape[0]
    fevd_ipc_data = fevd_summary[:, ipc_idx, :]

fevd_ipc = pd.DataFrame(
    fevd_ipc_data,
    columns=ENDOGENOUS,
    index=range(1, n_periods_fevd + 1)
)
fevd_ipc.index.name = 'Horizonte (meses)'
print(f"\n   FEVD para {IPC_COL}:")
print(fevd_ipc.round(4).to_string())

# ─── Exportar resultados ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("EXPORTANDO RESULTADOS")
print("=" * 80)

# Granger
granger_file = os.path.join(OUTPUT_DIR, 'granger_tests.csv')
df_granger.to_csv(granger_file, index=False, encoding='utf-8-sig')
print(f"   ✓ {granger_file}")

# ADF
adf_file = os.path.join(OUTPUT_DIR, 'adf_tests.csv')
pd.DataFrame(adf_results).to_csv(adf_file, index=False, encoding='utf-8-sig')
print(f"   ✓ {adf_file}")

# FEVD tabla (IPC)
fevd_file = os.path.join(OUTPUT_DIR, 'fevd_ipc.csv')
fevd_ipc.to_csv(fevd_file, encoding='utf-8-sig')
print(f"   ✓ {fevd_file}")

# FEVD para todas las variables
for var_idx, var_name in enumerate(ENDOGENOUS):
    if shape[0] == n_vars and shape[1] > shape[0]:
        fevd_var_data = fevd_summary[var_idx, :, :]
    else:
        fevd_var_data = fevd_summary[:, var_idx, :]
    fevd_var = pd.DataFrame(
        fevd_var_data,
        columns=ENDOGENOUS,
        index=range(1, n_periods_fevd + 1)
    )
    fevd_var.index.name = 'Horizonte (meses)'
    safe_name = var_name.replace(' ', '_').replace('/', '-')[:30]
    fevd_var_file = os.path.join(OUTPUT_DIR, f'fevd_{safe_name}.csv')
    fevd_var.to_csv(fevd_var_file, encoding='utf-8-sig')
print(f"   ✓ FEVD exportado para {len(ENDOGENOUS)} variables")

# IRF data export (JSON)
irf_data = {}
for i, impulse_var in enumerate(ENDOGENOUS):
    for j, response_var in enumerate(ENDOGENOUS):
        key = f"{impulse_var} -> {response_var}"
        irf_data[key] = irf.orth_irfs[:, j, i].tolist()

irf_file = os.path.join(OUTPUT_DIR, 'irf_data.json')
with open(irf_file, 'w', encoding='utf-8') as f:
    json.dump(irf_data, f, ensure_ascii=False, indent=2)
print(f"   ✓ {irf_file}")

# Resumen JSON
summary = {
    'var_order': optimal_lag,
    'aic_lag': int(optimal_lag_aic),
    'bic_lag': int(optimal_lag_bic),
    'n_obs': len(df_model),
    'variables': ENDOGENOUS,
    'differenced': bool(needs_diff),
    'differenced_vars': needs_diff,
    'significant_granger_pairs': [
        {'cause': r['cause'], 'effect': r['effect'], 'lag': r['lag'], 'p': r['f_pvalue']}
        for _, r in sig_granger.iterrows()
    ],
    'portmanteau_pvalue': float(durbin_watson),
    'fevd_ipc_horizon12': {k: round(v, 4) for k, v in fevd_ipc.iloc[-1].to_dict().items()}
}

json_file = os.path.join(OUTPUT_DIR, 'var_granger_summary.json')
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"   ✓ {json_file}")

print("\n" + "=" * 80)
print("✓ ANÁLISIS VAR/GRANGER COMPLETO")
print(f"   VAR({optimal_lag}) estimado con {len(df_model)} observaciones")
print(f"   Causalidades Granger significativas: {len(sig_granger)}")
print("=" * 80)
