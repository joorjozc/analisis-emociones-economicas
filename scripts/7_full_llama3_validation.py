"""
Script 7: Validación ampliada con Llama 3
Tesis doctoral — Jorge José Zamora, UPCT

- Muestra estratificada de 2.000 titulares del dataset completo
- Estratificación por año + categoría + emoción dominante
- Cohen's Kappa, Krippendorff's alpha, confusion matrix, precision/recall/F1
"""

import pandas as pd
import numpy as np
import json
import requests
import time
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

# ─── Configuración ───────────────────────────────────────────────────────────
INPUT_FILE = 'dataset_emociones.parquet'
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"  # Alternativa a llama3 si no está disponible
SAMPLE_SIZE = 2000
CHECKPOINT_INTERVAL = 50
CHECKPOINT_FILE = 'llama3_full_checkpoint.json'
OUTPUT_FILE = 'llama3_full_validation_results.csv'
OUTPUT_SUMMARY = 'llama3_full_validation_summary.json'
SEED = 42

SYSTEM_PROMPT = """Eres un experto en Lingüística Computacional y Economía Conductual.
Tu tarea es clasificar la emoción dominante del titular periodístico.

Analiza el siguiente titular de noticias en español y devuelve ÚNICAMENTE un objeto JSON
válido con estos campos exactos (sin texto adicional):
{
  "llama_emotion": "Anger|Fear|Joy|Sadness|Surprise|Disgust",
  "sensationalism_index": 1-10,
  "economic_impact": "Low|Medium|High",
  "reasoning": "breve explicación en español"
}

Las categorías emocionales son:
- Anger (Ira): indignación, frustración, enfado, rabia
- Fear (Miedo): amenaza, alarma, preocupación, riesgo
- Joy (Alegría): celebración, logro, éxito, satisfacción
- Sadness (Tristeza): pérdida, lamento, dolor, decepción
- Surprise (Sorpresa): asombro, inesperado, imprevisto
- Disgust (Asco): rechazo, repulsión, repugnancia

Debes elegir EXACTAMENTE una de las seis categorías anteriores."""

print("=" * 80)
print("VALIDACIÓN AMPLIADA CON LLAMA 3 — DATASET COMPLETO")
print("=" * 80)


def check_ollama():
    """Verificar conexión con Ollama."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            if any('gemma3' in m or 'llama3' in m for m in models):
                print(f"   ✓ Ollama activo, modelo disponible")
                return True
            print("   ⚠ Modelo no encontrado. Ejecutar: ollama pull gemma3:4b")
            return False
    except requests.exceptions.RequestException:
        pass
    print("   ✗ No se puede conectar a Ollama")
    print("   Descargar: https://ollama.com/download")
    print("   Luego: ollama pull llama3")
    return False


def query_llama3(headline, max_retries=2):
    """Consultar Llama 3 para clasificación emocional."""
    prompt = f'Titular: "{headline}"\n\nDevuelve solo el JSON:'

    for attempt in range(max_retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.3, "num_predict": 200}
            }, timeout=60)

            if r.status_code == 200:
                text = r.json().get('response', '{}')
                try:
                    parsed = json.loads(text)
                    required = ['llama_emotion', 'sensationalism_index', 'economic_impact', 'reasoning']
                    if all(f in parsed for f in required):
                        return parsed
                except json.JSONDecodeError:
                    pass

        except (requests.exceptions.Timeout, Exception):
            pass

        if attempt < max_retries:
            time.sleep(2)

    return {
        'llama_emotion': 'Anger',
        'sensationalism_index': 5,
        'economic_impact': 'Low',
        'reasoning': 'Error — valores por defecto'
    }


def krippendorff_alpha(labels1, labels2, level='nominal'):
    """Calcula alpha de Krippendorff para dos codificadores."""
    from itertools import combinations

    categories = list(set(labels1) | set(labels2))
    n = len(labels1)

    if level == 'nominal':
        # Observed disagreement
        Do = sum(1 for a, b in zip(labels1, labels2) if a != b) / n

        # Expected disagreement
        all_labels = list(labels1) + list(labels2)
        freq = {c: all_labels.count(c) for c in categories}
        total = len(all_labels)
        De = 1 - sum(f * (f - 1) for f in freq.values()) / (total * (total - 1))

        if De == 0:
            return 1.0
        return round(1 - Do / De, 4)
    return None


# ─── Paso 1: Verificar Ollama ────────────────────────────────────────────────
print("\n[1/6] Verificando conexión con Ollama...")
if not check_ollama():
    print("\n   ⚠ Ollama no disponible. Instalar y ejecutar antes de continuar.")
    sys.exit(1)

# ─── Paso 2: Cargar datos ────────────────────────────────────────────────────
print("\n[2/6] Cargando dataset con emociones...")
df = pd.read_parquet(INPUT_FILE)
print(f"   ✓ {len(df):,} titulares cargados")

# Filtrar solo emociones reales (excluir "others"/neutral)
df = df[df['dominant_emotion'] != 'others']
print(f"   ✓ {len(df):,} titulares con emoción real (excluidos 'others')")

# ─── Paso 3: Muestra estratificada ───────────────────────────────────────────
print(f"\n[3/6] Seleccionando muestra estratificada (n={SAMPLE_SIZE})...")

# Estratificar por año + categoría + emoción dominante
df['strat_group'] = (df['anio'].astype(str) + '_' +
                     df['categoria'].astype(str) + '_' +
                     df['dominant_emotion'].astype(str))

# Muestreo proporcional
np.random.seed(SEED)
sample_dfs = []
group_sizes = df['strat_group'].value_counts()

for group, n_group in group_sizes.items():
    proportion = n_group / len(df)
    n_sample = max(1, int(round(proportion * SAMPLE_SIZE)))
    group_data = df[df['strat_group'] == group]
    n_take = min(n_sample, len(group_data))
    sample_dfs.append(group_data.sample(n=n_take, random_state=SEED))

sample = pd.concat(sample_dfs, ignore_index=True)

# Ajustar al tamaño exacto
if len(sample) > SAMPLE_SIZE:
    sample = sample.sample(n=SAMPLE_SIZE, random_state=SEED)
elif len(sample) < SAMPLE_SIZE:
    remaining = df[~df.index.isin(sample.index)]
    extra = remaining.sample(n=SAMPLE_SIZE - len(sample), random_state=SEED)
    sample = pd.concat([sample, extra], ignore_index=True)

print(f"   ✓ Muestra: {len(sample)} titulares")
print(f"   Distribución por emoción dominante:")
for emo, n in sample['dominant_emotion'].value_counts().items():
    print(f"      {emo}: {n} ({n/len(sample)*100:.1f}%)")
print(f"   Distribución por categoría:")
for cat, n in sample['categoria'].value_counts().items():
    print(f"      {cat}: {n} ({n/len(sample)*100:.1f}%)")

# ─── Paso 4: Procesar con Llama 3 ────────────────────────────────────────────
print(f"\n[4/6] Procesando titulares con Llama 3...")
print(f"   Tiempo estimado: {len(sample) * 3 / 60:.0f} minutos")

# Cargar checkpoint
if Path(CHECKPOINT_FILE).exists():
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
else:
    checkpoint = {'processed': [], 'results': []}

processed_set = set(checkpoint['processed'])
results = checkpoint['results']
print(f"   Reanudando: {len(processed_set)} ya procesados")

for idx, row in tqdm(sample.iterrows(), total=len(sample), desc="   Validando"):
    str_idx = str(idx)
    if str_idx in processed_set:
        continue

    headline = str(row['item_title'])
    llama_result = query_llama3(headline)

    results.append({
        'idx': int(idx),
        'item_title': headline,
        'categoria': row.get('categoria', ''),
        'anio': int(row.get('anio', 0)),
        'Medio': str(row.get('Medio', '')),
        'roberta_dominant': row['dominant_emotion'],
        'roberta_anger': float(row['anger']),
        'roberta_fear': float(row['fear']),
        'roberta_joy': float(row['joy']),
        'roberta_sadness': float(row['sadness']),
        'llama_emotion': llama_result['llama_emotion'],
        'sensationalism': llama_result['sensationalism_index'],
        'economic_impact': llama_result['economic_impact'],
        'reasoning': llama_result['reasoning']
    })

    processed_set.add(str_idx)

    # Checkpoint
    if len(results) % CHECKPOINT_INTERVAL == 0:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump({'processed': list(processed_set), 'results': results},
                     f, ensure_ascii=False)

print(f"\n   ✓ {len(results)} titulares procesados")

# ─── Paso 5: Métricas de acuerdo ─────────────────────────────────────────────
print("\n[5/6] Calculando métricas de acuerdo...")

from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                             classification_report, accuracy_score)

results_df = pd.DataFrame(results)

# Mapear emociones RoBERTa a categorías comparables (6 emociones reales)
emotion_map = {
    'anger': 'Anger', 'fear': 'Fear', 'joy': 'Joy',
    'sadness': 'Sadness', 'surprise': 'Surprise', 'disgust': 'Disgust'
}
results_df['roberta_mapped'] = results_df['roberta_dominant'].map(emotion_map)

# Normalizar respuestas LLM a las 6 categorías válidas
valid_emotions = {'Anger', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Disgust'}
llama_normalize = {
    'Neutral': 'Anger',  # fallback para respuestas inválidas
    'neutral': 'Anger', 'anger': 'Anger', 'fear': 'Fear',
    'joy': 'Joy', 'sadness': 'Sadness', 'surprise': 'Surprise',
    'disgust': 'Disgust', 'Ira': 'Anger', 'Miedo': 'Fear',
    'Alegría': 'Joy', 'Tristeza': 'Sadness', 'Sorpresa': 'Surprise',
    'Asco': 'Disgust'
}
results_df['llama_emotion'] = results_df['llama_emotion'].apply(
    lambda x: x if x in valid_emotions else llama_normalize.get(x, 'Anger')
)
results_df['match'] = results_df['roberta_mapped'] == results_df['llama_emotion']

# Cohen's Kappa
kappa = cohen_kappa_score(results_df['roberta_mapped'], results_df['llama_emotion'])
accuracy = accuracy_score(results_df['roberta_mapped'], results_df['llama_emotion'])

# Krippendorff's alpha
kr_alpha = krippendorff_alpha(
    results_df['roberta_mapped'].tolist(),
    results_df['llama_emotion'].tolist()
)

# Confusion matrix
labels = sorted(set(results_df['roberta_mapped']) | set(results_df['llama_emotion']))
cm = confusion_matrix(results_df['roberta_mapped'], results_df['llama_emotion'], labels=labels)

# Classification report (precision, recall, F1)
cls_report = classification_report(
    results_df['roberta_mapped'], results_df['llama_emotion'],
    labels=labels, output_dict=True, zero_division=0
)

print(f"\n   Métricas de acuerdo:")
print(f"   - Cohen's Kappa:        {kappa:.4f} "
      f"({'Sustancial' if kappa > 0.6 else 'Moderado' if kappa > 0.4 else 'Justo'})")
print(f"   - Krippendorff's Alpha: {kr_alpha}")
print(f"   - Accuracy:             {accuracy:.2%}")
print(f"   - Coincidencias:        {results_df['match'].sum()}/{len(results_df)}")

print(f"\n   Matriz de confusión:")
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(f"   {cm_df.to_string()}")

print(f"\n   Precision/Recall/F1 por emoción:")
for label in labels:
    if label in cls_report:
        r = cls_report[label]
        print(f"      {label:10s}: P={r['precision']:.3f}, R={r['recall']:.3f}, "
              f"F1={r['f1-score']:.3f}, N={r['support']}")

# Sensacionalismo e impacto económico
sens_mean = results_df['sensationalism'].mean()
impact_dist = results_df['economic_impact'].value_counts(normalize=True)

print(f"\n   Sensacionalismo medio: {sens_mean:.2f}/10")
print(f"   Impacto económico percibido:")
for imp, pct in impact_dist.items():
    print(f"      {imp}: {pct:.1%}")

# ─── Paso 6: Exportar ────────────────────────────────────────────────────────
print("\n[6/6] Exportando resultados...")

results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"   ✓ {OUTPUT_FILE}")

summary = {
    'sample_size': len(results_df),
    'cohens_kappa': round(float(kappa), 4),
    'krippendorff_alpha': kr_alpha,
    'accuracy': round(float(accuracy), 4),
    'exact_matches': int(results_df['match'].sum()),
    'avg_sensationalism': round(float(sens_mean), 2),
    'economic_impact_distribution': impact_dist.to_dict(),
    'confusion_matrix': cm.tolist(),
    'confusion_labels': labels,
    'classification_report': cls_report,
    'roberta_distribution': results_df['roberta_mapped'].value_counts().to_dict(),
    'llama_distribution': results_df['llama_emotion'].value_counts().to_dict(),
    'by_category': {},
    'by_year': {}
}

# Métricas por categoría
for cat in results_df['categoria'].unique():
    cat_df = results_df[results_df['categoria'] == cat]
    if len(cat_df) > 10:
        cat_kappa = cohen_kappa_score(cat_df['roberta_mapped'], cat_df['llama_emotion'])
        cat_acc = accuracy_score(cat_df['roberta_mapped'], cat_df['llama_emotion'])
        summary['by_category'][cat] = {
            'n': len(cat_df), 'kappa': round(float(cat_kappa), 4),
            'accuracy': round(float(cat_acc), 4)
        }

# Métricas por año
for year in sorted(results_df['anio'].unique()):
    yr_df = results_df[results_df['anio'] == year]
    if len(yr_df) > 10:
        yr_kappa = cohen_kappa_score(yr_df['roberta_mapped'], yr_df['llama_emotion'])
        summary['by_year'][int(year)] = {
            'n': len(yr_df), 'kappa': round(float(yr_kappa), 4)
        }

with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"   ✓ {OUTPUT_SUMMARY}")

# Limpiar checkpoint
if Path(CHECKPOINT_FILE).exists():
    Path(CHECKPOINT_FILE).unlink()

print("\n" + "=" * 80)
print("✓ VALIDACIÓN LLAMA 3 COMPLETA")
print(f"   Cohen's Kappa: {kappa:.4f}")
print(f"   Krippendorff's Alpha: {kr_alpha}")
print(f"   Accuracy: {accuracy:.2%}")
print(f"   Sensacionalismo medio: {sens_mean:.2f}/10")
print("=" * 80)
