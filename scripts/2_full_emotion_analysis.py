"""
Script 2: Clasificación emocional completa con RoBERTa-base-bne
Tesis doctoral — Jorge José Zamora, UPCT

Procesa el dataset completo (253K titulares) usando el pipeline de transformers
directamente (más rápido que pysentimiento.predict por menos overhead).
Optimizado para CPU: checkpoints, batch_size alto, reanudación automática.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ─── Configuración ───────────────────────────────────────────────────────────
INPUT_FILE = 'dataset_limpio.parquet'
OUTPUT_FILE = 'dataset_emociones.parquet'
CHECKPOINT_DIR = 'checkpoints_emotions'
CHUNK_SIZE = 10000       # procesar y guardar en chunks de 10K
BATCH_SIZE = 64          # batch para el pipeline de transformers
SEED = 42

EMOTION_COLS = ['others', 'joy', 'sadness', 'anger', 'surprise', 'disgust', 'fear']

print("=" * 80)
print("CLASIFICACIÓN EMOCIONAL COMPLETA — RoBERTa-base-bne")
print("=" * 80)

# ─── Paso 1: Cargar datos ────────────────────────────────────────────────────
print("\n[1/4] Cargando dataset limpio...")
df = pd.read_parquet(INPUT_FILE)
total = len(df)
headlines = df['item_title'].fillna('').astype(str).tolist()
print(f"   ✓ {total:,} titulares cargados")

# ─── Paso 2: Preparar checkpoints ────────────────────────────────────────────
print("\n[2/4] Preparando sistema de checkpoints...")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

progress_file = os.path.join(CHECKPOINT_DIR, 'progress.json')
start_chunk = 0

if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    start_chunk = progress.get('last_chunk', 0)
    print(f"   ✓ Reanudando desde chunk {start_chunk} "
          f"({start_chunk * CHUNK_SIZE:,}/{total:,} = "
          f"{start_chunk * CHUNK_SIZE / total * 100:.1f}%)")
else:
    print(f"   ✓ Inicio desde cero")

# ─── Paso 3: Inicializar pipeline ────────────────────────────────────────────
print("\n[3/4] Inicializando pipeline de transformers...")
print("   Modelo: pysentimiento/robertuito-emotion-analysis")

from transformers import pipeline as hf_pipeline

classifier = hf_pipeline(
    'text-classification',
    model='pysentimiento/robertuito-emotion-analysis',
    top_k=None,         # devuelve todas las probabilidades
    device=-1,           # CPU
    batch_size=BATCH_SIZE,
    truncation=True,
    max_length=128
)
print("   ✓ Pipeline cargado y listo")

# ─── Paso 4: Procesar por chunks ─────────────────────────────────────────────
n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"\n[4/4] Procesando {total:,} titulares en {n_chunks} chunks de {CHUNK_SIZE:,}...")
print(f"   Batch size del pipeline: {BATCH_SIZE}")

t_start = time.time()
errors = 0

# Preallocar arrays para resultados
all_emotions = np.zeros((total, 7), dtype=np.float32)
all_dominant = [''] * total

for chunk_idx in range(start_chunk, n_chunks):
    chunk_start = chunk_idx * CHUNK_SIZE
    chunk_end = min(chunk_start + CHUNK_SIZE, total)
    chunk_headlines = headlines[chunk_start:chunk_end]
    chunk_size_actual = len(chunk_headlines)

    t_chunk = time.time()

    try:
        # Pipeline procesa internamente en batches de BATCH_SIZE
        results = classifier(chunk_headlines)

        for i, result_list in enumerate(results):
            row_idx = chunk_start + i
            # result_list es una lista de dicts [{'label': 'anger', 'score': 0.1}, ...]
            probs = {item['label']: item['score'] for item in result_list}

            for j, col in enumerate(EMOTION_COLS):
                all_emotions[row_idx, j] = probs.get(col, 0.0)

            # Emoción dominante
            all_dominant[row_idx] = max(probs, key=probs.get)

    except Exception as e:
        errors += 1
        print(f"\n   ⚠ Error en chunk {chunk_idx} ({chunk_start}-{chunk_end}): {e}")
        # Intentar uno por uno como fallback
        for i, headline in enumerate(chunk_headlines):
            try:
                result_list = classifier(headline)[0]
                row_idx = chunk_start + i
                probs = {item['label']: item['score'] for item in result_list}
                for j, col in enumerate(EMOTION_COLS):
                    all_emotions[row_idx, j] = probs.get(col, 0.0)
                all_dominant[row_idx] = max(probs, key=probs.get)
            except Exception:
                pass

    # Guardar checkpoint
    with open(progress_file, 'w') as f:
        json.dump({
            'last_chunk': chunk_idx + 1,
            'total_chunks': n_chunks,
            'total': total,
            'errors': errors,
            'elapsed_seconds': time.time() - t_start
        }, f)

    # Guardar chunk parcial
    chunk_file = os.path.join(CHECKPOINT_DIR, f'chunk_{chunk_idx:04d}.npz')
    np.savez_compressed(chunk_file,
                        emotions=all_emotions[chunk_start:chunk_end],
                        dominant=all_dominant[chunk_start:chunk_end])

    # Progreso
    elapsed = time.time() - t_start
    chunk_time = time.time() - t_chunk
    processed = chunk_end
    rate = processed / elapsed if elapsed > 0 else 0
    remaining = (total - processed) / rate if rate > 0 else 0

    print(f"   Chunk {chunk_idx+1}/{n_chunks} "
          f"({processed:,}/{total:,} = {processed/total*100:.1f}%) "
          f"— {chunk_time:.0f}s/chunk — "
          f"ETA: {remaining/60:.0f}min — "
          f"Vel: {rate:.0f} tit/s")

# ─── Paso 5: Guardar resultado final ─────────────────────────────────────────
elapsed_total = time.time() - t_start
print(f"\n   ✓ Procesamiento completado en {elapsed_total/3600:.1f} horas")
print(f"   Errores: {errors}")

print(f"\n   Guardando {OUTPUT_FILE}...")
for j, col in enumerate(EMOTION_COLS):
    df[col] = all_emotions[:, j]
df['dominant_emotion'] = all_dominant

df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"   ✓ Guardado: {OUTPUT_FILE} ({file_size:.1f} MB)")

# ─── Estadísticas resumen ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ESTADÍSTICAS RESUMEN")
print("=" * 80)

print("\nProbabilidades medias por emoción:")
for col in EMOTION_COLS:
    print(f"   {col:12s}: media={df[col].mean():.4f}, std={df[col].std():.4f}")

print("\nDistribución de emoción dominante:")
dom_dist = df['dominant_emotion'].value_counts()
for emotion, count in dom_dist.items():
    print(f"   {emotion:12s}: {count:,} ({count/total*100:.1f}%)")

print(f"\n   Tiempo total: {elapsed_total/3600:.1f} horas")
print(f"   Velocidad media: {total/elapsed_total:.0f} titulares/segundo")
print("=" * 80)
