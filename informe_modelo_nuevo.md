# Informe: un modelo distinto al de referencia, en los dos sprints

Análisis de Datos · Ingeniería de Sistemas · 2026-2

**Encargo:** crear un modelo nuevo y diferente al de referencia y aplicarlo al taller. Referencias: regresión lineal + duración² (sprint 1) y regresión logística (sprint 2).

**Resumen.** Los dos modelos nuevos son de otra familia, superan con holgura la línea base, pasan `pytest` y la prueba de humo, y quedan empatados con la referencia: **ninguno la supera**. Es un resultado esperable en estos datos y se explica abajo.

| | Sprint 1 (regresión) | Sprint 2 (clasificación) |
|---|---|---|
| Modelo nuevo | Aditivo: splines cúbicos + Ridge | Análisis discriminante lineal (LDA) con encogimiento |
| Referencia | Lineal + duración² | Logística |
| En prueba 2020-2025 | MAE **5,82** | Costo **109** |
| Referencia en prueba | MAE 5,75 | Costo 99 |
| Línea base | MAE 10,44 | Costo 318 |
| Mejora frente a la línea base | −44 % del error | −66 % del costo |
| Archivo | `models/modelo.pkl` | `models/modelo_sprint2.pkl` |

## 1. Por qué no se esperaba una mejora

Antes de modelar se buscó estructura que el modelo lineal no capturara, usando solo 1995-2019 y residuos fuera de muestra: las medias de los residuos son planas en todas las variables, la dispersión es constante entre géneros, colaboración y años, y la mayor correlación de un residuo con cualquier interacción candidata es 0,08 (ruido para unas 1.270 filas). Es decir, los datos son casi lineales con ruido de unos 7,6 puntos, y la referencia ya está cerca del mejor MAE alcanzable (≈ 6). Un modelo más flexible solo puede igualarla.

## 2. Sprint 1: modelo aditivo (splines + Ridge)

**Qué es.** Cada variable continua (bailabilidad, energía, valencia, acústica, tempo, duración, volumen) pasa por una curva suave (spline cúbico, 4 nudos); año y colaboración entran lineales (el año debe extrapolarse a 2026 en adelante); género en indicadoras. Ridge (alpha = 1) evita que las curvas se ajusten al ruido. La duración ya no necesita el término cuadrático a mano: la curva en U la aprende sola (máximo cerca de 3,3 min).

**Regla (fijada antes de ver resultados).** Candidatos de otras familias; elegir el de menor MAE en la validación interna de ventana creciente (4 pliegues dentro de 1995-2019); hiperparámetros elegidos allí, nunca con la prueba; informar aparte si supera a la referencia (mejora ≥ 0,05 y gana en 3 de 4 pliegues).

| Corrida | MAE interno | MAE en prueba | RMSE | R² |
|---|---|---|---|---|
| Media por género (línea base) | – | 10,44 | 13,16 | 0,05 |
| Lineal + duración² (referencia) | 6,06 | **5,75** | 7,15 | 0,718 |
| **Aditivo: splines + Ridge** | 6,16 | **5,82** | 7,25 | 0,710 |
| Ridge con todas las interacciones | 6,81 | 5,95 | 7,42 | 0,697 |
| Huber (pérdida robusta) | 6,06 | 5,75 | 7,15 | 0,718 |

**Una decisión que debo declarar.** Aplicada al pie de la letra, mi regla eligió Huber (menor MAE interno). Pero sus predicciones difieren de las de la referencia solo 0,13 puntos en promedio (correlación 0,9999): es la misma recta con otra función de pérdida, no un modelo distinto, y no cumpliría el encargo. Elegí entonces el aditivo, que es el candidato verdaderamente distinto (sus predicciones difieren 0,6 puntos en promedio). Es una decisión tomada **después** de ver los resultados y por eso la dejo escrita.

**Frente a la referencia.** Mejora interna −0,10 (peor), gana en 0 de 4 pliegues: no supera la regla del taller. En prueba, la diferencia pareada de error absoluto (referencia − nuevo) es −0,07 puntos con IC95 % bootstrap [−0,15; +0,01]: el intervalo incluye 0, así que es un **empate estadístico**. Las curvas aprendidas casi coinciden con las rectas de la referencia (figura `m1_curvas_y_diagnostico.png`), con una leve saturación de energía en valores altos: confirma que la relación es esencialmente lineal.

**Diagnóstico del modelo aplicado (2020-2025).** MAE 5,82 y RMSE 7,25 (RMSE/MAE = 1,25: hay errores grandes, pero no dominan); el 51 % queda a ±5 puntos y el 81 % a ±10 (la referencia: 84 %). Sesgo global +0,29; por género, la mayor subestimación es pop (+1,48) y la mayor sobreestimación, indie (−0,86); el mayor error está en urbano (MAE 6,24) y el menor en rock (5,23). Con corte 65 se promocionan 147 canciones y se capturan 90 de los 106 hits (16 quedan sin promoción).

## 3. Sprint 2: análisis discriminante lineal (LDA)

**Qué es.** Un clasificador generativo: supone que, dentro de cada clase (hit / no hit), las variables se distribuyen como gaussianas con la misma covarianza (encogida automáticamente), y obtiene la probabilidad de hit con la regla de Bayes. Es de otra familia que la logística, que modela directamente la probabilidad. Umbral elegido con el costo (falso negativo = 3, falso positivo = 1) sobre probabilidades fuera de muestra.

**Regla (corrige un hueco de mi análisis anterior).** En un primer intento solo miraba el costo, y eso dejó pasar una logística balanceada cuya «mejora» era un cambio de umbral con probabilidades infladas (esos archivos se conservan: `models/modelo_sprint2_balanceada.pkl` y `sprint_2/`). Ahora, además del costo interno, un candidato solo es elegible si su log-loss interno no supera en más de 5 % el de la referencia, para que las probabilidades sigan siendo probabilidades.

| Corrida | Umbral | Costo interno | Log-loss interno | Costo en prueba | Precisión | Recall | ROC-AUC | Prob. media prevista |
|---|---|---|---|---|---|---|---|---|
| Siempre «no hit» | – | 774 | – | 318 | – | 0,00 | 0,50 | – |
| Logística (referencia) | 0,27 | 322 | 0,309 | **99** | 0,688 | 0,811 | 0,918 | 0,236 |
| Logística aditiva (splines) | 0,26 | 351 | 0,326 | 109 | 0,637 | 0,811 | 0,916 | 0,240 |
| Logística aditiva calibrada | 0,43 | 356 | 0,388 | 119 | 0,525 | 0,896 | 0,910 | 0,414 |
| **LDA** | 0,30 | **320** | **0,303** | **109** | 0,675 | 0,783 | 0,913 | 0,241 |

Las dos logísticas aditivas no pasan la regla (peor costo y log-loss). LDA es elegible y tiene el menor costo interno. **Lectura honesta:** en la validación interna casi empata con la referencia (320 frente a 322, gana en 2 de 4 pliegues) y en la prueba cuesta 10 puntos más (109 frente a 99). Esa diferencia equivale a unos pocos errores de 415 canciones; no hay evidencia de que LDA sea mejor, pero tampoco de que sea mucho peor.

**Lo que sí aporta: probabilidades que se pueden leer.** En prueba, la probabilidad media prevista es 0,24 y la tasa real de hits 0,26, y la curva de calibración sigue la diagonal (los cinco grupos tienen tasas reales 0,00; 0,01; 0,17; 0,33 y 0,77 frente a probabilidades medias 0,01; 0,05; 0,14; 0,31 y 0,70). Por eso el campo `probabilidad_hit` del servicio es interpretable y el umbral teórico FP/(FP+FN) = 0,25 también funciona (costo 106). Con el umbral por defecto 0,50 el costo sube a 156; con el elegido, 109; el mejor posible viendo la prueba (no utilizable) sería 104.

**Sensibilidad al costo del falso negativo** (umbral reelegido en validación interna): con costo 2, 97 frente a 212 de la línea base; con 3, 109 frente a 318; con 5, 149 frente a 530. **Por género (prueba):** indie es donde más hits se pierden (recall 0,43, con solo 7 hits) y rock donde más promoción se desperdicia (precisión 0,47); pop concentra el mayor costo (48).

## 4. Conclusión y límites

Dos modelos de familias distintas a las de referencia, aplicados al servicio, que reducen un 44 % el error (sprint 1) y un 66 % el costo (sprint 2) frente a la línea base. No mejoran a las referencias porque los datos no tienen estructura que una familia más flexible pueda aprovechar; inventar una mejora no sería honesto. **Límites:** datos sintéticos, un solo periodo de prueba, asociaciones y no causas; en el sprint 2 los costos 3 y 1 los debe confirmar quien responde por el presupuesto; en ambos sprints los hiperparámetros se eligieron con la validación interna y eso introduce un optimismo leve en esas cifras (la prueba, no).

## 5. Entregables

`modelo_nuevo/` (scripts de análisis, registros `registro_sprint1_nuevo.csv` y `registro_sprint2_nuevo.csv`, resultados `.json`, figuras `m1` y `m2`), `src/variables.py` (se añadió `preprocesamiento_aditivo`, sin modificar lo existente), `src/entrenar_regresion.py` y `src/entrenar_clasificacion.py` ajustados, `models/modelo.pkl` (sprint 1), `models/modelo_sprint2.pkl` (sprint 2) y las referencias conservadas en `models/modelo_sprint1_referencia.pkl` y `models/modelo_sprint2_balanceada.pkl`.
