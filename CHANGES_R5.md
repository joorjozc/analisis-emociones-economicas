# Revisión de septiembre de 2026 (scripts 10-14)

Los revisores del artículo derivado de la tesis (IEEE Access, Access-2026-17671) objetaron tres cosas que
el análisis de la tesis compartía: la causalidad de Granger se contrastaba par a par eligiendo el «mejor
rezago» sin corrección por comparaciones múltiples; el orden del VAR no estaba justificado; y no se
comprobaba si la curación de Google News afecta a los resultados. Al aplicar esos criterios a los datos de
la tesis (2014-2024, 132 meses) aparecieron además dos errores propios. Todo lo que sigue se calcula con
los scripts nuevos a partir de `results/econometric_dataset_global.csv` (y, en el script 11, de los
titulares, que no se distribuyen); las salidas están en `results/r5/`.

## Qué cambia

- **Orden del VAR** (`10_r5_var_granger_fevd.py`). VAR(6) se mantiene, ahora con una regla explícita:
  es el menor orden con residuos blancos (Portmanteau a p + 5 rezagos; VAR(2)-VAR(5) fallan, VAR(6)
  p = 0,275). El script reproduce primero la FEVD y el Portmanteau publicados en `results/robustness/`.
- **Granger** (script 10). Los pares bivariantes con «mejor rezago» (`results/var_granger/granger_tests.csv`)
  no sobreviven a ninguna corrección. Se sustituyen por (a) tests conjuntos de bloque: las cuatro
  emociones → IPC, p = 0,002 (Holm 0,003; bootstrap residual con 1.999 réplicas, 0,004), y IPC → emociones,
  p = 0,022; y (b) los 20 tests par a par dentro del VAR(6) con q-valores de Benjamini-Hochberg: ninguno
  queda por debajo de 0,05 (`results/r5/tabla_7_14_pares.csv`).
- **FEVD** (script 10). El texto de la tesis leía `fevd.decomp` con los ejes traspuestos (statsmodels lo
  indexa como ecuación, horizonte, shock) y atribuía a las emociones cuotas del 95-98 % que eran las cuotas
  propias de cada variable. Corregido: a 12 meses las emociones explican el 25,8 % de la varianza del IPC
  (Cholesky, ordenación de referencia; 5 ordenaciones en `tabla_fevd_ordenaciones.csv`) y el 33,1 % con la
  FEVD generalizada normalizada por filas, con el miedo en cabeza (13,1 %).
- **Quiebres estructurales** (`11_r5_breaks_curation.py`). El sup-F de Bai-Perron se compara con valores
  críticos simulados (10.000 simulaciones; la F estándar no vale, Andrews 1993). El IPC solo rompe en las
  mismas fechas. Se construye un panel fijo de 229 medios presentes todos los años (94,6 % de los
  titulares) con pesos fijos: los quiebres, la caída de la alegría y el test conjunto emociones → IPC se
  mantienen; la subida de la ira en la década baja del 34,2 % al 20,6 % (`fixed_composition_emotions.csv`).
- **Índice de negatividad** (`12_r5_negativity_index.py`). Ira + miedo + tristeza en un VAR con IPC y
  alegría: anticipa el IPC solo a seis rezagos (p = 0,003; panel fijo 0,011), no con 1-5 rezagos ni bajo
  Toda-Yamamoto; la respuesta acumulada no tiene signo.
- **ARIMAX** (`13_r5_arimax_multiplicity.py`). Corrección BH/Holm por familias sobre los coeficientes de
  `results/arimax/all_results.json`: de 14 nominales sobreviven 5 (`arimax_multiplicity.csv`). El RMSE y el
  MAE de `4_full_arimax_models.py` incluían el residuo de la inicialización difusa (≈ nivel de la serie);
  recalculados sin él (`arimax_rmse_corrected.csv`; IPC general 8,11 → 0,63) y Figura 7.11 rehecha.
- **Cointegración** (`14_r5_cointegration_toda_yamamoto.py`). El estadístico de traza de Johansen
  publicado (10,68) usa cinco diferencias retardadas; con 1-4 rechaza la ausencia de cointegración
  (`johansen_sensitivity.csv`). Se añade Toda-Yamamoto (VAR en niveles con p + 1 rezagos, Wald sobre los p
  primeros): emociones → IPC no se sostiene (p = 0,15); IPC → emociones sí (p = 0,008).

## Qué no cambia
Corpus, clasificación con RoBERTa-base-bne, validación con Gemma 3, perfiles por categoría, tests de Chow
y Kruskal-Wallis, y los scripts 1-9 con sus resultados, que se conservan tal cual como referencia.

## Ubicación
Este repositorio sustituye a https://github.com/Jorgejosezamora/analisis-emociones-economicas, cuya
cuenta ya no es accesible. Conserva el historial completo de commits.
