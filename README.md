# Análisis del sentimiento en medios digitales españoles y su influencia en el comportamiento del consumidor (2014-2024)

**Tesis doctoral** — Jorge José Zamora Cánovas
Universidad Politécnica de Cartagena (UPCT), España

## Descripción

Este repositorio contiene el código, las figuras y la documentación de una tesis doctoral que analiza la relación entre el contenido emocional de los medios digitales españoles y los indicadores de comportamiento del consumidor durante el periodo pre- y pospandemia (2014-2024).

La investigación se articula sobre tres pilares teóricos: la **teoría del framing** (cómo los encuadres mediáticos configuran la percepción pública de los eventos económicos), el **agenda-setting** (la capacidad de los medios para influir en los temas que la ciudadanía considera relevantes) y el **PLN aplicado a la economía** (uso de modelos de lenguaje basados en transformadores para el análisis de sentimiento a gran escala del discurso económico).

El estudio abarca una década que incluye la recuperación de la crisis financiera de 2008, la pandemia de COVID-19 y el periodo inflacionario posterior, lo que proporciona una ventana única para examinar cómo el tono emocional de los medios interactúa con la dinámica macroeconómica en España.

**Dataset**: 253.886 titulares procedentes de 1.436 medios digitales españoles (2014-2024), recopilados mediante feeds RSS de Google News.

## Metodología

El análisis sigue un pipeline de tres etapas:

1. **Clasificación emocional** — RoBERTa-base-bne (PlanTL-GOB-ES) clasifica cada titular en 7 categorías emocionales: ira, miedo, alegría, tristeza, sorpresa, asco y otros (neutral).
2. **Modelización econométrica** — Los modelos ARIMAX (con selección de orden via `auto_arima` y re-estimación con `SARIMAX`) y VAR examinan la relación temporal entre las emociones mediáticas agregadas y los indicadores macroeconómicos (IPC-ECOICOP, Índice de Confianza del Consumidor, afiliaciones a la Seguridad Social).
3. **Validación cruzada** — Gemma 3:4b (vía Ollama) clasifica de forma independiente 16.651 titulares no neutrales para validar las asignaciones emocionales del modelo principal.

Los análisis complementarios incluyen pruebas de causalidad de Granger, funciones de impulso-respuesta, descomposición de varianza (FEVD estándar y generalizada Pesaran-Shin), detección de cambios estructurales en torno a la pandemia de COVID-19 y comprobaciones de robustez (tests de cointegración, sensibilidad al orden de rezagos VAR y al ordenamiento Cholesky en la FEVD).

## Resultados principales

| Hipótesis | Hallazgo |
|---|---|
| **H1**: Las emociones mediáticas influyen en los indicadores macroeconómicos | Parcialmente confirmada — significativa para el IPC, limitada para el ICC y las afiliaciones a la SS |
| **H2**: Las emociones negativas predominan en la cobertura económica | Confirmada — ira (6.347) y tristeza (5.208) superan a la alegría (3.273) entre los titulares no neutrales |
| **H3**: La COVID-19 produjo un cambio estructural en los patrones emocionales | Confirmada — las pruebas de Bai-Perron y Chow detectan cambio de régimen en marzo de 2020 |
| **H4**: La validación cruzada con LLM respalda las clasificaciones primarias | Confirmada — concordancia sustancial entre RoBERTa y Gemma 3 |

## Estructura del repositorio

```
├── scripts/                        # Pipeline de análisis
│   ├── cod_tesis.R                 # Script R para recopilación de titulares (Google News RSS)
│   ├── 1_dataset_census.py         # Limpieza y censo del dataset
│   ├── 2_full_emotion_analysis.py  # Clasificación emocional con RoBERTa
│   ├── 3_monthly_aggregation.py    # Agregación mensual + indicadores macro
│   ├── 4_full_arimax_models.py     # Modelos econométricos ARIMAX
│   ├── 5_var_granger_analysis.py   # Modelos VAR y causalidad de Granger
│   ├── 6_structural_breaks.py      # Detección de cambios estructurales
│   ├── 7_full_llama3_validation.py # Validación cruzada con LLM (Gemma 3:4b)
│   ├── 8_generate_thesis_figures.py
│   ├── 9_robustness_checks.py     # Comprobaciones de robustez (cointegración, FEVD, VAR)
│   ├── 10_r5_var_granger_fevd.py  # Revisión 2026: VAR(6), Granger conjunto + BH, FEVD corregida, IRF
│   ├── 11_r5_breaks_curation.py   # Revisión 2026: sup-F simulado, panel fijo de medios
│   ├── 12_r5_negativity_index.py  # Revisión 2026: índice de negatividad como predictor
│   ├── 13_r5_arimax_multiplicity.py # Revisión 2026: BH sobre los ARIMAX y RMSE corregido
│   ├── 14_r5_cointegration_toda_yamamoto.py # Revisión 2026: Johansen (sensibilidad) y Toda-Yamamoto
│   └── r5_common.py                # Funciones compartidas (bootstrap de Granger, GFEVD)
├── data/                           # Datos de entrada (no incluidos — ver data/README.md)
├── results/                        # Resultados de todos los scripts (incluidos)
│   ├── arimax/                     # Modelos ARIMAX, pruebas ADF, gráficos de diagnóstico
│   ├── var_granger/                # Resultados VAR, pruebas de Granger, IRF, FEVD
│   ├── structural_breaks/          # Pruebas de Bai-Perron, Chow, comparaciones de régimen
│   ├── robustness/                 # Cointegración, robustez VAR lag-order, FEVD ordering, GFEVD
│   ├── validation/                 # Validación cruzada RoBERTa vs Gemma 3
│   ├── r5/                         # Resultados de la revisión de 2026 (scripts 10-14)
│   ├── econometric_dataset_*.csv   # Datasets mensuales agregados
│   ├── volume_dataset.csv          # Volumen de titulares por fuente
│   └── censo_report.json           # Estadísticas del censo del dataset
├── figures/                        # 27 figuras de la tesis (PDF + PNG, ver más abajo)
│   └── r5/                         # Figuras rehechas en la revisión de 2026 (7.11, 7.13, 7.14, 7.23)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Figuras

Todas las figuras están disponibles en `figures/` en formato PDF y PNG.

| Figura | Descripción |
|--------|-------------|
| `fig_6_1` | Evolución temporal del volumen de titulares (2014-2024) |
| `fig_6_2` | Diagrama del pipeline metodológico |
| `fig_7_1` | Volumen mensual de titulares |
| `fig_7_3` | Distribución de emociones dominantes (no neutrales, N = 16.651) |
| `fig_7_4` | Boxplots de puntuaciones de confianza por emoción |
| `fig_7_5` | Evolución temporal de las emociones |
| `fig_7_6` | Media móvil de 12 meses de las emociones |
| `fig_7_7` | Índice de negatividad (ira + miedo + tristeza) |
| `fig_7_8` | Perfil emocional por periodo (pre-COVID / COVID / post-COVID) |
| `fig_7_9` | Mapa de calor de emociones por categoría mediática |
| `fig_7_10` | Tendencias por categoría mediática |
| `fig_7_25` | Robustez FEVD bajo distintos ordenamientos Cholesky + GFEVD Pesaran-Shin |
| `fig_7_26` | Robustez del modelo VAR según orden de rezagos (VAR(2), VAR(4), VAR(6)) |
| `r5/fig_7_11_arimax_global` | ARIMAX del IPC general, observado frente a ajustado (sin el residuo de inicialización) |
| `r5/fig_7_13_irf` | Funciones de impulso-respuesta ortogonalizadas del VAR(6) con bandas al 95 % |
| `r5/fig_7_14_fevd` | Descomposición de la varianza del IPC (Cholesky y generalizada), VAR(6) |
| `r5/fig_7_23_granger_pares` | Contrastes de Granger par a par con q-valores de Benjamini-Hochberg |

## Tecnologías

| Componente | Herramienta |
|------------|-------------|
| Clasificación emocional | [RoBERTa-base-bne](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne) (PlanTL-GOB-ES) |
| Validación cruzada con LLM | [Gemma 3:4b](https://ollama.com/library/gemma3) vía [Ollama](https://ollama.com/) |
| Modelización econométrica | Python · statsmodels · pmdarima |
| Recopilación de datos | R · Google News RSS |
| Visualización | matplotlib · seaborn |
| Procesamiento de datos | pandas · NumPy · scikit-learn |

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/joorjozc/analisis-emociones-economicas.git
cd analisis-emociones-economicas

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

Coloca los archivos de datos necesarios en el directorio de trabajo (consulta `data/README.md` para más detalles) y ejecuta los scripts de forma secuencial:

```bash
cd scripts
python 1_dataset_census.py
python 2_full_emotion_analysis.py
python 3_monthly_aggregation.py
python 4_full_arimax_models.py
python 5_var_granger_analysis.py
python 6_structural_breaks.py
python 7_full_llama3_validation.py    # Requiere Ollama con Gemma 3:4b
python 8_generate_thesis_figures.py
python 9_robustness_checks.py         # Comprobaciones de robustez
python 10_r5_var_granger_fevd.py      # Revisión de 2026 (ver CHANGES_R5.md)
python 11_r5_breaks_curation.py       # Requiere los titulares (HEADLINES_PARQUET)
python 12_r5_negativity_index.py
python 13_r5_arimax_multiplicity.py
python 14_r5_cointegration_toda_yamamoto.py
```

> **Nota**: El Script 2 (~253K titulares) y el Script 7 (~16,6K titulares vía LLM) son de larga ejecución. Ambos soportan reanudación mediante checkpoints.

> **Nota**: El Script 7 requiere [Ollama](https://ollama.com/) con el modelo `gemma3:4b` instalado localmente.

## Revisión de septiembre de 2026

Los scripts 10-14 revisan la parte econométrica con los criterios que exigió la revisión por pares del artículo derivado de la tesis: tests conjuntos de Granger con corrección por comparaciones múltiples (Holm, bootstrap, Benjamini-Hochberg), regla explícita de selección del orden del VAR, FEVD generalizada (Pesaran-Shin), valores críticos simulados para el sup-F de Bai-Perron, un panel fijo de medios frente a la curación de Google News, el índice de negatividad como predictor y contrastes de Toda-Yamamoto. Cada script reproduce primero las cifras publicadas en `results/` como control. Los cambios y sus consecuencias están en [`CHANGES_R5.md`](CHANGES_R5.md).

Este repositorio sustituye a `github.com/Jorgejosezamora/analisis-emociones-economicas`, cuya cuenta ya no es accesible; conserva su historial completo.

## Fuentes de datos

- **Titulares**: Google News RSS (medios españoles, 2014-2024)
- **IPC (ECOICOP)**: [INE — Instituto Nacional de Estadística](https://www.ine.es/)
- **Índice de Confianza del Consumidor**: [CIS — Centro de Investigaciones Sociológicas](https://www.cis.es/)
- **Afiliaciones a la Seguridad Social**: [MITES — Ministerio de Trabajo](https://www.mites.gob.es/)

## Cita

Si utilizas este código en tu investigación, por favor cita:

**APA 7:**

> Zamora Cánovas, J. J. (2026). *Análisis del sentimiento en medios digitales españoles y su influencia en el comportamiento del consumidor: una aproximación empírica pospandemia (2014-2024)* [Tesis doctoral, Universidad Politécnica de Cartagena].

**BibTeX:**

```bibtex
@phdthesis{zamora2026sentiment,
  title={Análisis del sentimiento en medios digitales españoles y su influencia
         en el comportamiento del consumidor: una aproximación empírica
         pospandemia (2014-2024)},
  author={Zamora Cánovas, Jorge José},
  year={2026},
  school={Universidad Politécnica de Cartagena}
}
```

También puedes usar el archivo `CITATION.cff` para herramientas de citación automática.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT — consulta [LICENSE](LICENSE) para más detalles.
