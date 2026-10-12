# Informe breve: predicción de popularidad de canciones (Semana 9)

Análisis de Datos · Ingeniería de Sistemas · 2026-2

## 1. Problema y momento de la decisión

Una disquera decide, **antes del lanzamiento**, qué canciones reciben inversión promocional. Se estima la popularidad (0-100) con `canciones.csv` (2.000 canciones sintéticas, 1995-2025). Solo son legítimas las variables que se conocen antes de publicar: atributos de audio, género, año y colaboración. Se entrena con 1995-2019 (1.585 canciones) y se prueba una sola vez con 2020-2025 (415).

## 2. Calidad de los datos

No hay nulos ni duplicados. Los valores extremos son pocos (menos de 1 % por columna) y plausibles (tempo 60-200 bpm, volumen hasta 0 dB, duración 1,5-6,15 min), por lo que se conservan; reajustar sin las 21 filas con algún extremo deja el error en prueba igual (5,754 frente a 5,756). Las explicativas casi no se correlacionan entre sí (VIF ≈ 1), así que no hay redundancia que obligue a retirar variables.

## 3. Preparación de variables y control de fuga

| Columna | Decisión | Evidencia |
|---|---|---|
| `reproducciones_sem1` | Fuera | Ocurre una semana después del lanzamiento. Correlación 0,99 con el objetivo. |
| `es_hit` | Fuera | Es «popularidad ≥ 70» escrita de otra forma (correlación 0,67). |
| Resto de columnas | Entran | Existen antes del lanzamiento. |

Todo preprocesamiento (codificación de género y escalado) se ajusta dentro del pipeline, solo con entrenamiento.

## 4. Modelado y registro comparativo

El registro completo (14 corridas, incluidas las descartadas) está en `registro_semana9.csv`. Resumen sobre las mismas 415 canciones de 2020-2025:

| Corrida | MAE | RMSE | R² | Lectura |
|---|---|---|---|---|
| Media global | 11,06 | 13,77 | −0,04 | Sin modelo |
| Media por género | 10,44 | 13,16 | 0,05 | Línea base |
| Lineal, todas las columnas | 1,41 | 1,97 | 0,98 | Con fuga: no válido |
| Lineal, columnas válidas | 5,87 | 7,36 | 0,70 | Resultado honesto |
| Lineal + duración² | **5,75** | 7,15 | 0,72 | **Seleccionado** |
| Gradient boosting | 6,66 | 8,29 | 0,62 | Subestima (sesgo +2,45) |
| Random Forest | 7,13 | 8,88 | 0,57 | Más complejo no es mejor |

**Regla de selección (fijada antes de ver resultados):** se elige con la validación interna de ventana creciente (4 pliegues dentro de entrenamiento); una variante más compleja reemplaza a la simple solo si mejora el MAE interno en al menos 0,05 y gana en 3 de 4 pliegues. `Duración²` la cumple (MAE interno de 6,26 a 6,06, gana en 4 de 4). Mi **idea propia**, `clima = bailabilidad × valencia` (una canción bailable y alegre podría ganar más que la suma de ambos efectos), da 6,05 frente a 6,06: una mejora de 0,006, dentro del ruido, que no cumple la regla. Se registra y se descarta por parsimonia, igual que la interacción de clase `bailabilidad × energía`. Con la partición aleatoria 80/20 el modelo lineal obtiene casi el mismo error (5,83), es decir, en estos datos no fue más optimista que la temporal.

## 5. Métricas y diagnóstico

El modelo reduce el MAE de la línea base en 45 % (de 10,44 a 5,75 puntos). El RMSE (7,15) no es mucho mayor que el MAE (relación 1,24): hay errores grandes, pero no dominan. El 84 % de las canciones queda a ±10 puntos de lo esperado y el 51 % a ±5. El sesgo global es +0,26, pero por género cambia:

| Género | Canciones | MAE | Sesgo |
|---|---|---|---|
| Urbano | 98 | 6,03 | +0,42 |
| Pop | 128 | 5,95 | +1,32 |
| Indie | 53 | 5,68 | −0,85 |
| Electrónica | 69 | 5,64 | −0,12 |
| Rock | 67 | 5,14 | −0,71 |

Falla más en **urbano** (mayor MAE); el sesgo más claro es la **subestimación en pop** (+1,32) y la sobreestimación en indie (−0,85).

## 6. Importancia y conclusión de negocio

Por permutación, pesan sobre todo `bailabilidad` (+4,4 puntos de MAE al barajarla), `energia` (+2,9) y `genero` (+1,6); `tempo`, `anio` y `duracion_min` pesan poco. Son **asociaciones, no causas**: el coeficiente negativo de indie (−5,3) no significa que el género cause menor popularidad, porque también decide listas e inversión.

**Conclusión.** El modelo sirve para **priorizar** el catálogo (las medias por decil coinciden con la diagonal), no para pronosticar con precisión una canción individual: una de cada seis queda a más de 10 puntos. Con presupuesto para unas 150 canciones, un corte de 65 puntos promociona 148 canciones y captura 90 de los 106 hits de 2020-2025, dejando 16 sin promoción; cortes más altos promocionan menos pero pierden más hits (corte 70: 87 promocionadas y 38 hits sin promoción). El corte lo debe fijar quien responde por el presupuesto y pondera el costo de cada error. **Límites:** datos sintéticos, un solo periodo de prueba y asociaciones sin interpretación causal.

## 7. Auditoría de la salida del asistente de IA

El asistente afirmó R² = 0,98, «precisión del 98 %», variables más importantes `reproducciones_sem1` y `es_hit`, «generaliza bien» y «listo para canciones nuevas». Se aplicó el protocolo de cinco preguntas, cambiando una sola cosa por paso sobre su modelo:

| Afirmación | Evidencia ejecutable | Veredicto |
|---|---|---|
| «R² de 0,98» | Con semilla fija da 0,979 en dos corridas idénticas | Reproducible |
| «Precisión del 98 %» | R² es varianza explicada, no aciertos; en el modelo corregido solo 51 % queda a ±5 puntos | Falso |
| «Variables más importantes: reproducciones y es_hit» | Ninguna existe antes del lanzamiento; sin ellas R² = 0,69 y MAE sube de 1,39 a 6,16 | Fuga |
| «Generaliza bien» | Solo probó al azar; con 2020-2025 y sin fuga: R² 0,70, MAE 5,87 | Compatible con la evidencia; la afirmación no estaba sustentada |
| «Listo para canciones nuevas» | No comparó con línea base (10,44) ni dijo su error en puntos | Sin sustento |

**Memo (cinco líneas)**

1. El resultado del asistente (R² = 0,98) se reproduce, pero usa dos columnas que no existen antes del lanzamiento: `reproducciones_sem1` y `es_hit`.
2. Sin esas columnas el error pasa de 1,39 a 6,16 puntos; el R² honesto, con partición temporal, es 0,70 y no 0,98.
3. Contra la línea base (10,44 puntos) el modelo corregido y mejorado (lineal + duración², 5,75) la reduce en 45 %: es útil para priorizar, no para pronosticar una canción.
4. «Precisión del 98 %», «generaliza bien» y «listo para canciones nuevas» no estaban sustentadas.
5. Recomendación: usar el modelo corregido con su ficha y darle al asistente el momento de la decisión como parte del encargo.

## 8. Entregables

`semana_09/Notebook_DA_S9_Regresion_Canciones.ipynb`, `registro_semana9.csv`, este informe, `ficha_auditoria.csv` y `memo_auditoria.txt`; en el repositorio, `models/modelo.pkl` (con su ficha) y `src/entrenar_regresion.py` ajustado. El servicio local pasa `pytest -q` (9 pruebas) y la prueba de humo («servicio en orden»). La publicación en Render y la URL quedan pendientes: se hacen desde la cuenta propia (ver pasos en el README del repositorio).
