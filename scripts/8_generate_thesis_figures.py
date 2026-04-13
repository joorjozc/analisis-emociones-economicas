"""
Script 8: Generación de figuras para la tesis (Capítulos 6 y 7)
Tesis doctoral — Jorge José Zamora, UPCT

Genera todas las figuras en calidad publicación (300dpi, PDF+PNG).
Estilo consistente, compatible con escala de grises.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
import os
import json
import warnings

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# ─── Configuración de estilo ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Paleta compatible con escala de grises
COLORS = {
    'Ira': '#d62728',      'Miedo': '#9467bd',
    'Alegría': '#2ca02c',   'Tristeza': '#1f77b4',
    'Sorpresa': '#ff7f0e',  'Asco': '#8c564b',
    'Otros': '#7f7f7f',     'Neutral': '#7f7f7f'
}
GRAY_HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o']

INPUT_GLOBAL = 'econometric_dataset_global.csv'
INPUT_BY_CAT = 'econometric_dataset_by_category.csv'
INPUT_VOLUME = 'volume_dataset.csv'
INPUT_EMOTIONS = 'dataset_emociones.parquet'
INPUT_CENSO = 'censo_report.json'

OUTPUT_DIR = 'figuras_tesis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTION_VARS_ES = ['Ira', 'Miedo', 'Alegría', 'Tristeza', 'Sorpresa', 'Asco']


def save_fig(fig, name):
    """Guardar figura en PNG y PDF."""
    png_path = os.path.join(OUTPUT_DIR, f'{name}.png')
    pdf_path = os.path.join(OUTPUT_DIR, f'{name}.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ {name} (.png + .pdf)")


print("=" * 80)
print("GENERACIÓN DE FIGURAS PARA LA TESIS")
print("=" * 80)

# ─── Cargar datos ─────────────────────────────────────────────────────────────
print("\n[0] Cargando datos...")
df_global = pd.read_csv(INPUT_GLOBAL, parse_dates=['Fecha'])
df_global = df_global.sort_values('Fecha')

df_by_cat = pd.read_csv(INPUT_BY_CAT, parse_dates=['Fecha'])

if os.path.exists(INPUT_EMOTIONS):
    df_emotions = pd.read_parquet(INPUT_EMOTIONS)
    print(f"   ✓ Dataset emociones: {len(df_emotions):,}")
else:
    df_emotions = None
    print("   ⚠ dataset_emociones.parquet no encontrado — algunas figuras se omitirán")

if os.path.exists(INPUT_CENSO):
    with open(INPUT_CENSO, 'r', encoding='utf-8') as f:
        censo = json.load(f)
    print(f"   ✓ Censo: {censo['total_clean']:,} titulares")
else:
    censo = None

print(f"   ✓ Dataset global: {len(df_global)} obs")
print(f"   ✓ Dataset por categoría: {len(df_by_cat)} filas")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS CAPÍTULO 6
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FIGURAS CAPÍTULO 6 — METODOLOGÍA")
print("=" * 80)

# ─── Figura 6.1: Línea temporal 2014-2024 con eventos clave ──────────────────
print("\n[6.1] Línea temporal con eventos clave...")
fig, ax = plt.subplots(figsize=(14, 3))

events = [
    ('2014-01', 'Inicio\ncorpus', 'black'),
    ('2016-06', 'Brexit', '#d62728'),
    ('2017-10', 'Crisis\ncatalana', '#ff7f0e'),
    ('2020-03', 'COVID-19', '#d62728'),
    ('2022-02', 'Guerra\nUcrania', '#ff7f0e'),
    ('2024-12', 'Fin\ncorpus', 'black'),
]

ax.axhline(0, color='gray', lw=2)
for date, label, color in events:
    x = pd.Timestamp(date + '-01')
    ax.axvline(x, color=color, ls='--', lw=1.5, alpha=0.7)
    ax.annotate(label, xy=(x, 0.5), fontsize=9, ha='center', va='bottom',
                fontweight='bold', color=color)

ax.set_xlim(pd.Timestamp('2013-06-01'), pd.Timestamp('2025-06-01'))
ax.set_ylim(-0.5, 1.5)
ax.set_yticks([])
ax.set_title('Figura 6.1. Línea temporal del estudio (2014-2024)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
save_fig(fig, 'fig_6_1_linea_temporal')

# ─── Figura 6.2: Pipeline de agregación ──────────────────────────────────────
print("\n[6.2] Pipeline de agregación (diagrama esquemático)...")
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis('off')

boxes = [
    (1, 1.5, 'Dataset\n~1M titulares', '#e6f2ff'),
    (3, 1.5, 'RoBERTa\n7 emociones', '#fff2e6'),
    (5, 1.5, 'Agregación\nmensual', '#e6ffe6'),
    (7, 1.5, 'Variables\neconómicas', '#ffe6e6'),
    (9, 1.5, 'Dataset\neconométrico', '#f2e6ff'),
]

for x, y, text, color in boxes:
    ax.add_patch(plt.Rectangle((x-0.7, y-0.5), 1.4, 1, fill=True,
                                facecolor=color, edgecolor='black', lw=1.5))
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

for i in range(len(boxes) - 1):
    ax.annotate('', xy=(boxes[i+1][0]-0.7, boxes[i+1][1]),
                xytext=(boxes[i][0]+0.7, boxes[i][1]),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

ax.set_title('Figura 6.2. Pipeline: del corpus al dataset econométrico', fontsize=13)
save_fig(fig, 'fig_6_2_pipeline')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS CAPÍTULO 7
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FIGURAS CAPÍTULO 7 — RESULTADOS")
print("=" * 80)

# ─── Figura 7.1: Volumen mensual de titulares ────────────────────────────────
print("\n[7.1] Volumen mensual de titulares...")
if 'n_titulares' in df_global.columns:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_global['Fecha'], df_global['n_titulares'], color='steelblue', lw=1.5)
    ax.fill_between(df_global['Fecha'], df_global['n_titulares'], alpha=0.2, color='steelblue')
    ax.axvline(pd.Timestamp('2020-03-01'), color='red', ls='--', lw=1, label='COVID-19')
    ax.axvline(pd.Timestamp('2022-02-01'), color='orange', ls='--', lw=1, label='Ucrania')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Número de titulares')
    ax.set_title('Figura 7.1. Volumen mensual de titulares (2014-2024)')
    ax.legend()
    save_fig(fig, 'fig_7_1_volumen_mensual')

# ─── Figura 7.3: Distribución de emociones dominantes ────────────────────────
print("\n[7.3] Distribución de emociones dominantes (excluyendo neutral)...")
if df_emotions is not None and 'dominant_emotion' in df_emotions.columns:
    dom_counts = df_emotions['dominant_emotion'].value_counts()
    dom_counts = dom_counts.drop('others', errors='ignore')
    n_real = dom_counts.sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    color_map_en = {'anger': '#d62728', 'fear': '#9467bd', 'joy': '#2ca02c',
                    'sadness': '#1f77b4', 'surprise': '#ff7f0e', 'disgust': '#8c564b'}
    bars = ax.bar(range(len(dom_counts)), dom_counts.values,
                  color=[color_map_en.get(e, '#7f7f7f') for e in dom_counts.index])
    ax.set_xticks(range(len(dom_counts)))
    ax.set_xticklabels([e.capitalize() for e in dom_counts.index], rotation=45, ha='right')
    ax.set_ylabel('Número de titulares')
    ax.set_title('Figura 7.3. Distribución de emociones dominantes (excluyendo neutral)')
    for i, (val, name) in enumerate(zip(dom_counts.values, dom_counts.index)):
        ax.text(i, val + max(dom_counts)*0.01, f'{val:,}\n({val/n_real*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    save_fig(fig, 'fig_7_3_emociones_dominantes')

# ─── Figura 7.4: Box plots de probabilidades emocionales ─────────────────────
print("\n[7.4] Box plots de probabilidades (excluyendo neutral)...")
if df_emotions is not None:
    emotion_cols_en = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust']
    available = [c for c in emotion_cols_en if c in df_emotions.columns]
    if available:
        color_map_bp = {'anger': '#d62728', 'fear': '#9467bd', 'joy': '#2ca02c',
                        'sadness': '#1f77b4', 'surprise': '#ff7f0e', 'disgust': '#8c564b'}
        # Muestra para boxplot (todo el dataset es demasiado para boxplot)
        sample_bp = df_emotions[available].sample(min(50000, len(df_emotions)), random_state=42)
        fig, ax = plt.subplots(figsize=(12, 6))
        bp = ax.boxplot([sample_bp[c].values for c in available],
                       labels=[c.capitalize() for c in available],
                       patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], [color_map_bp.get(c, '#7f7f7f') for c in available]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Probabilidad')
        ax.set_title('Figura 7.4. Distribución de probabilidades emocionales (excluyendo neutral)')
        save_fig(fig, 'fig_7_4_boxplots_emociones')

# ─── Figura 7.5: Evolución mensual de las 7 emociones ────────────────────────
print("\n[7.5] Evolución mensual de emociones (figura central)...")
fig, ax = plt.subplots(figsize=(16, 7))
for emo in EMOTION_VARS_ES:
    if emo in df_global.columns:
        ax.plot(df_global['Fecha'], df_global[emo], label=emo,
                color=COLORS.get(emo, 'gray'), lw=1.5)

ax.axvline(pd.Timestamp('2020-03-01'), color='red', ls='--', lw=1, alpha=0.5, label='COVID-19')
ax.axvline(pd.Timestamp('2022-02-01'), color='orange', ls='--', lw=1, alpha=0.5, label='Ucrania')
ax.set_xlabel('Fecha')
ax.set_ylabel('Probabilidad media mensual')
ax.set_title('Figura 7.5. Evolución mensual de las emociones en titulares (2014-2024)')
ax.legend(ncol=4, loc='upper left')
save_fig(fig, 'fig_7_5_evolucion_emociones')

# ─── Figura 7.6: Media móvil 12 meses ────────────────────────────────────────
print("\n[7.6] Tendencia suavizada (media móvil 12 meses)...")
fig, ax = plt.subplots(figsize=(16, 7))
for emo in EMOTION_VARS_ES:
    if emo in df_global.columns:
        ma12 = df_global[emo].rolling(12, min_periods=6).mean()
        ax.plot(df_global['Fecha'], ma12, label=emo,
                color=COLORS.get(emo, 'gray'), lw=2)

ax.axvline(pd.Timestamp('2020-03-01'), color='red', ls='--', lw=1, alpha=0.5)
ax.axvline(pd.Timestamp('2022-02-01'), color='orange', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('Fecha')
ax.set_ylabel('Probabilidad media (MA-12)')
ax.set_title('Figura 7.6. Tendencia suavizada de emociones (media móvil 12 meses)')
ax.legend(ncol=4, loc='upper left')
save_fig(fig, 'fig_7_6_media_movil_12m')

# ─── Figura 7.7: Índice compuesto de negatividad ─────────────────────────────
print("\n[7.7] Índice compuesto de negatividad...")
if 'Negatividad' in df_global.columns:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_global['Fecha'], df_global['Negatividad'], color='darkred', lw=1.5)
    ax.fill_between(df_global['Fecha'], df_global['Negatividad'], alpha=0.2, color='red')
    ma12 = df_global['Negatividad'].rolling(12, min_periods=6).mean()
    ax.plot(df_global['Fecha'], ma12, color='black', lw=2, ls='--', label='MA-12')
    ax.axvline(pd.Timestamp('2020-03-01'), color='red', ls='--', lw=1, alpha=0.5)
    ax.axvline(pd.Timestamp('2022-02-01'), color='orange', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Negatividad (Ira + Miedo + Tristeza)')
    ax.set_title('Figura 7.7. Índice compuesto de negatividad (2014-2024)')
    ax.legend()
    save_fig(fig, 'fig_7_7_negatividad')

# ─── Figura 7.8: Perfil emocional por período ────────────────────────────────
print("\n[7.8] Perfil emocional por período...")
periods = {
    'Pre-COVID\n(2014-2019)': (df_global['Fecha'] < '2020-03-01'),
    'COVID\n(2020-2021)': (df_global['Fecha'] >= '2020-03-01') & (df_global['Fecha'] < '2022-01-01'),
    'Post-COVID\n(2022-2024)': (df_global['Fecha'] >= '2022-01-01')
}

fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(EMOTION_VARS_ES))
width = 0.25

for i, (period_name, mask) in enumerate(periods.items()):
    means = [df_global.loc[mask, emo].mean() if emo in df_global.columns else 0
             for emo in EMOTION_VARS_ES]
    ax.bar(x_pos + i * width, means, width, label=period_name, alpha=0.8)

ax.set_xticks(x_pos + width)
ax.set_xticklabels(EMOTION_VARS_ES)
ax.set_ylabel('Probabilidad media')
ax.set_title('Figura 7.8. Perfil emocional por período')
ax.legend()
save_fig(fig, 'fig_7_8_perfil_periodos')

# ─── Figura 7.9: Heatmap emociones × categorías ──────────────────────────────
print("\n[7.9] Heatmap emociones × categorías...")
cats = df_by_cat['categoria'].unique()
heatmap_data = []
for cat in sorted(cats):
    row = {}
    for emo in EMOTION_VARS_ES:
        if emo in df_by_cat.columns:
            row[emo] = df_by_cat[df_by_cat['categoria'] == cat][emo].mean()
    row['Categoría'] = cat
    heatmap_data.append(row)

df_heat = pd.DataFrame(heatmap_data).set_index('Categoría')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_heat, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Probabilidad media'})
ax.set_title('Figura 7.9. Perfil emocional por categoría temática')
save_fig(fig, 'fig_7_9_heatmap_categorias')

# ─── Figuras 7.10a-f: Tendencias por categoría ───────────────────────────────
print("\n[7.10] Tendencias temporales por categoría...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
axes_flat = axes.flatten()

for i, cat in enumerate(sorted(cats)[:6]):
    ax = axes_flat[i]
    cat_data = df_by_cat[df_by_cat['categoria'] == cat].sort_values('Fecha')
    for emo in ['Ira', 'Miedo', 'Alegría', 'Tristeza']:
        if emo in cat_data.columns:
            ax.plot(cat_data['Fecha'], cat_data[emo], label=emo,
                    color=COLORS.get(emo, 'gray'), lw=1)
    ax.set_title(f'{cat}', fontweight='bold')
    ax.axvline(pd.Timestamp('2020-03-01'), color='red', ls='--', lw=0.5, alpha=0.5)
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle('Figuras 7.10a-f. Evolución emocional por categoría temática', fontsize=14)
plt.tight_layout()
save_fig(fig, 'fig_7_10_tendencias_categorias')

# ─── Figuras de robustez ────────────────────────────────────────────────────
ROBUSTNESS_DIR = 'robustness_results'

# Figura 7.25: Robustez del ordenamiento FEVD
robustness_fevd_file = os.path.join(ROBUSTNESS_DIR, 'fevd_ordering_table.csv')
if os.path.exists(robustness_fevd_file):
    print("\n[7.25] Robustez del ordenamiento FEVD (Cholesky + Generalizada)...")
    df_fevd_rob = pd.read_csv(robustness_fevd_file)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(df_fevd_rob))
    width = 0.15
    cols = ['CPI', 'Ira', 'Miedo', 'Tristeza', 'Alegría']
    col_labels = cols.copy()
    col_labels[0] = 'IPC'
    bar_colors = ['#1f77b4', '#d62728', '#9467bd', '#1f77b4', '#2ca02c']

    for i, (col, label) in enumerate(zip(cols, col_labels)):
        if col in df_fevd_rob.columns:
            ax.bar(x + i * width, df_fevd_rob[col], width, label=label,
                   color=bar_colors[i], alpha=0.8)

    ax.set_xlabel('Ordenamiento')
    ax.set_ylabel('Proporción de varianza (h=12)')
    ax.set_title('Figura 7.25. FEVD del IPC bajo distintos ordenamientos Cholesky y GFEVD')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_fevd_rob['Ordering'], rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_fig(fig, 'fig_7_25_fevd_ordering_robustness')
else:
    print("\n[7.25] ⚠ No se encontró fevd_ordering_table.csv — ejecutar script 9 primero")

# Figura 7.26: Robustez del orden de rezagos VAR
robustness_var_file = os.path.join(ROBUSTNESS_DIR, 'var_robustness.json')
if os.path.exists(robustness_var_file):
    print("\n[7.26] Robustez del orden de rezagos VAR...")
    with open(robustness_var_file, 'r', encoding='utf-8') as f:
        var_rob = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    lag_orders = []
    n_sig_pairs = []
    port_pvalues = []
    fevd_cpi_own = []

    for key, data in var_rob.items():
        if 'error' not in data:
            lag_orders.append(data['lag_order'])
            n_sig_pairs.append(data['n_significant_granger_pairs'])
            port_pvalues.append(data['portmanteau_pvalue'])
            ipc_col_key = [k for k in data['fevd_cpi_horizon12'] if 'IPC' in k or 'Índice' in k]
            if ipc_col_key:
                fevd_cpi_own.append(data['fevd_cpi_horizon12'][ipc_col_key[0]])
            else:
                fevd_cpi_own.append(list(data['fevd_cpi_horizon12'].values())[0])

    x_labels = [f'VAR({l})' for l in lag_orders]

    axes[0].bar(x_labels, n_sig_pairs, color='steelblue', alpha=0.8)
    axes[0].set_title('Pares Granger significativos', fontweight='bold')
    axes[0].set_ylabel('N.º pares (p < 0.05)')

    axes[1].bar(x_labels, port_pvalues, color='coral', alpha=0.8)
    axes[1].axhline(0.05, color='red', ls='--', lw=1, label='α = 0.05')
    axes[1].set_title('Test Portmanteau (p-value)', fontweight='bold')
    axes[1].set_ylabel('p-value')
    axes[1].legend()

    axes[2].bar(x_labels, fevd_cpi_own, color='seagreen', alpha=0.8)
    axes[2].set_title('FEVD: varianza propia IPC (h=12)', fontweight='bold')
    axes[2].set_ylabel('Proporción')

    plt.suptitle('Figura 7.26. Robustez del modelo VAR según orden de rezagos',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig_7_26_var_lag_robustness')
else:
    print("\n[7.26] ⚠ No se encontró var_robustness.json — ejecutar script 9 primero")

# ─── Resumen ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
figures = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
print(f"✓ {len(figures)} FIGURAS GENERADAS EN {OUTPUT_DIR}/")
for f in sorted(figures):
    print(f"   - {f}")
print("=" * 80)
