"""
Script 1: Censo y limpieza del Dataset completo
Tesis doctoral — Jorge José Zamora, UPCT

Carga Dataset.xlsx (~1M filas), filtra España, elimina columnas AFINN,
deduplica y genera estadísticas descriptivas + dataset limpio en parquet.
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Configuración
INPUT_FILE = 'Dataset.xlsx'
OUTPUT_PARQUET = 'dataset_limpio.parquet'
OUTPUT_REPORT = 'censo_report.json'

AFINN_COLS = ['palabras_encontradas', 'suma_sentimientos', 'sentimiento_ponderado']

print("=" * 80)
print("CENSO Y LIMPIEZA DEL DATASET COMPLETO")
print("=" * 80)

# ─── Paso 1: Carga ───────────────────────────────────────────────────────────
print("\n[1/6] Cargando Dataset.xlsx (esto puede tardar varios minutos)...")
df = pd.read_excel(INPUT_FILE, sheet_name='Tratada', engine='openpyxl')
total_raw = len(df)
print(f"   ✓ Cargadas {total_raw:,} filas, {len(df.columns)} columnas")
print(f"   Columnas: {list(df.columns)}")

# ─── Paso 2: Filtrar España ──────────────────────────────────────────────────
print("\n[2/6] Filtrando País == 'España'...")
df_spain = df[df['País'] == 'España'].copy()
total_spain = len(df_spain)
total_other = total_raw - total_spain
print(f"   ✓ España: {total_spain:,} filas ({total_spain/total_raw*100:.1f}%)")
print(f"   Descartadas (otro país): {total_other:,}")

# ─── Paso 3: Descartar columnas AFINN ────────────────────────────────────────
print("\n[3/6] Descartando columnas AFINN...")
cols_before = list(df_spain.columns)
df_spain = df_spain.drop(columns=[c for c in AFINN_COLS if c in df_spain.columns])
cols_dropped = [c for c in AFINN_COLS if c in cols_before]
print(f"   ✓ Eliminadas: {cols_dropped}")
print(f"   Columnas finales ({len(df_spain.columns)}): {list(df_spain.columns)}")

# ─── Paso 4: Deduplicación ───────────────────────────────────────────────────
print("\n[4/6] Deduplicando por item_title...")
n_before = len(df_spain)
df_spain = df_spain.drop_duplicates(subset='item_title', keep='first')
n_after = len(df_spain)
n_dupes = n_before - n_after
print(f"   ✓ Duplicados eliminados: {n_dupes:,}")
print(f"   Filas finales: {n_after:,}")

# ─── Paso 5: Estadísticas descriptivas ───────────────────────────────────────
print("\n[5/6] Calculando estadísticas descriptivas...")

# Por categoría
cat_dist = df_spain['categoria'].value_counts().to_dict()
print("\n   Distribución por categoría:")
for cat, n in sorted(cat_dist.items(), key=lambda x: -x[1]):
    print(f"      {cat}: {n:,} ({n/n_after*100:.1f}%)")

# Por año
year_dist = df_spain['anio'].value_counts().sort_index().to_dict()
print("\n   Distribución por año:")
for y, n in year_dist.items():
    print(f"      {y}: {n:,} ({n/n_after*100:.1f}%)")

# Por mes
month_dist = df_spain['mes_pub'].value_counts().sort_index().to_dict()

# Top medios
medio_dist = df_spain['Medio'].value_counts().head(30).to_dict()
print("\n   Top 20 medios:")
for i, (m, n) in enumerate(list(medio_dist.items())[:20]):
    print(f"      {i+1}. {str(m).strip()}: {n:,}")

# Titulares por año-mes (para serie temporal)
df_spain['anio_mes'] = df_spain['anio'].astype(str) + '-' + df_spain['mes_pub'].astype(str).str.zfill(2)
monthly_counts = df_spain.groupby('anio_mes').size().to_dict()

# Valores faltantes
missing = df_spain.isnull().sum().to_dict()
missing_nonzero = {k: v for k, v in missing.items() if v > 0}

# ─── Paso 6: Exportar ────────────────────────────────────────────────────────
print(f"\n[6/6] Exportando a {OUTPUT_PARQUET}...")
df_spain.to_parquet(OUTPUT_PARQUET, index=False, engine='pyarrow')
file_size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)
print(f"   ✓ Guardado: {OUTPUT_PARQUET} ({file_size_mb:.1f} MB)")

# Guardar reporte JSON
report = {
    'total_raw': int(total_raw),
    'total_spain': int(total_spain),
    'total_other_country': int(total_other),
    'duplicates_removed': int(n_dupes),
    'total_clean': int(n_after),
    'columns': list(df_spain.columns),
    'n_columns': len(df_spain.columns),
    'distribution_by_category': {str(k): int(v) for k, v in cat_dist.items()},
    'distribution_by_year': {str(k): int(v) for k, v in year_dist.items()},
    'distribution_by_month': {str(k): int(v) for k, v in month_dist.items()},
    'top_30_media': {str(k).strip(): int(v) for k, v in medio_dist.items()},
    'monthly_counts': monthly_counts,
    'missing_values': {str(k): int(v) for k, v in missing_nonzero.items()},
    'unique_categories': int(df_spain['categoria'].nunique()),
    'unique_media': int(df_spain['Medio'].nunique()),
    'date_range': {
        'min_year': int(df_spain['anio'].min()),
        'max_year': int(df_spain['anio'].max())
    }
}

with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"   ✓ Reporte guardado: {OUTPUT_REPORT}")

print("\n" + "=" * 80)
print("✓ CENSO COMPLETO")
print("=" * 80)
print(f"\n   Total bruto:    {total_raw:,}")
print(f"   Filtro España:  {total_spain:,}")
print(f"   Deduplicado:    {n_after:,}")
print(f"   Categorías:     {df_spain['categoria'].nunique()}")
print(f"   Medios únicos:  {df_spain['Medio'].nunique()}")
print(f"   Período:        {df_spain['anio'].min()}-{df_spain['anio'].max()}")
print(f"\n   Archivos generados:")
print(f"   - {OUTPUT_PARQUET} ({file_size_mb:.1f} MB)")
print(f"   - {OUTPUT_REPORT}")
print("=" * 80)
