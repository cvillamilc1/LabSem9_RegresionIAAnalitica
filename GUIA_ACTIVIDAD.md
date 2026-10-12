# Guía de la actividad: de los datos al servicio en Render

Esta guía explica **qué pide el README, por qué, y qué se hizo en cada paso**. Está escrita para entenderla de principio a fin, no solo para copiar comandos.

---

## 1. Qué se pide, en una frase

Una disquera quiere decidir, **antes del lanzamiento**, en qué canciones invertir promoción. Hay que entrenar un modelo, guardarlo en `models/modelo.pkl` junto con su «ficha» y publicar un servicio en Render que responda predicciones. Se hace dos veces:

| | Sprint 1 | Sprint 2 |
|---|---|---|
| Pregunta | ¿Qué popularidad (0-100) tendrá? | ¿Será un hit (popularidad ≥ 70)? |
| Tipo de problema | Regresión | Clasificación binaria |
| Cómo se mide | MAE (error medio en puntos) | Costo de los errores |
| Decisión | Promocionar si la popularidad esperada supera 65 | Promocionar si la probabilidad de hit supera un umbral |

El código del servicio **no se toca**: lo único que cambia es el archivo `models/modelo.pkl`.

## 2. Cómo encajan las piezas del repositorio

- `data/canciones.csv`: 2.000 canciones sintéticas, 1995-2025.
- `src/variables.py`: define las 10 columnas de entrada, la variable derivada `duracion_c2` (duración centrada en 3,5 min y elevada al cuadrado), el preprocesamiento y la partición por año.
- `src/entrenar_regresion.py` y `src/entrenar_clasificacion.py`: entrenan, evalúan y llaman a `guardar()`.
- `src/artefacto.py`: `guardar()` valida la ficha, prueba que el pipeline prediga y escribe el `.pkl`; `cargar()` lo lee.
- `src/app/main.py`: servicio FastAPI. No tiene reglas propias: lee de la ficha la tarea, el corte o umbral y el error esperado.
- `tests/test_servicio.py` y `src/probar_servicio.py`: pruebas locales y prueba de humo contra la URL publicada (con esta se califica).

## 3. Conceptos que sostienen todo

**Fuga de datos (leakage).** Usar información que no existe en el momento de decidir. Aquí hay dos columnas trampa: `reproducciones_sem1` (ocurre después del lanzamiento y correlaciona 0,99 con la popularidad) y `es_hit`/`popularidad` (una es la respuesta escrita de otra forma). Con ellas el R² llega a 0,98, pero es un espejismo: sin ellas es 0,70.

**Partición temporal.** Se entrena con 1995-2019 y se prueba con 2020-2025, porque el modelo va a predecir canciones **futuras**. La prueba se usa **una sola vez**; si se usara para elegir, dejaría de ser una prueba.

**Validación interna de ventana creciente.** Para comparar modelos sin mirar la prueba, dentro de 1995-2019 se entrena con el pasado y se valida con el tramo siguiente, cuatro veces (`TimeSeriesSplit`).

**Pipeline.** Un solo objeto que deriva variables, codifica género (one-hot), escala números y estima. Todo lo que se «aprende» de los datos se ajusta solo con entrenamiento. Por eso el servicio puede recibir las 10 columnas crudas y funcionar.

**Línea base.** Lo mínimo que hay que vencer: en regresión, la media por género (MAE 10,44); en clasificación, «nunca promocionar» (costo 318). Un modelo que no la supera no sirve.

**MAE, RMSE, R², sesgo (regresión).** MAE: error típico en puntos. RMSE: castiga más los errores grandes. R²: fracción de varianza explicada, **no** porcentaje de aciertos. Sesgo: media de los residuos (positivo = subestima).

**Costo y umbral (clasificación).** No todos los errores cuestan igual: dejar sin promoción un hit (FN) cuesta 3 y promocionar un no-hit (FP) cuesta 1. El costo es `3·FN + 1·FP`. El umbral es la probabilidad a partir de la cual se promociona; se elige el que minimiza el costo con probabilidades **fuera de muestra** de la validación interna.

**La ficha.** El diccionario guardado dentro del `.pkl`: qué predice, con qué datos, qué variables se excluyeron y por qué, línea base, desempeño, decisión y límites. El endpoint `GET /modelo` la muestra.

## 4. Sprint 1: regresión (modelo nuevo)

Punto de partida: el repositorio traía el modelo de referencia (lineal + duración², MAE 5,75) y una carpeta `semana_09/` con un notebook ejecutado y su informe. Verifiqué que ese resultado se reproduce y **lo conservé** en `models/modelo_sprint1_referencia.pkl`.

El modelo nuevo es un **aditivo con splines + Ridge**: una curva suave por variable continua, año y colaboración lineales, género en indicadoras, y Ridge para no ajustar el ruido. Sus hiperparámetros (4 nudos, alpha = 1) se eligieron con validación interna temporal, no con la prueba.

- Resultado en 2020-2025: **MAE 5,82** (línea base 10,44, −44 %); R² 0,710.
- **No supera a la referencia** (5,75): la diferencia es un empate estadístico. Antes de modelar comprobé que los residuos del modelo lineal no tienen estructura, así que los datos no ofrecen nada que una familia más flexible pueda aprovechar.
- Declaré una decisión: mi regla eligió Huber, pero sus predicciones son casi idénticas a las de la referencia (no es un modelo distinto); por eso apliqué el aditivo.

## 5. Sprint 2: clasificación (modelo nuevo)

El modelo nuevo es un **análisis discriminante lineal (LDA)** con encogimiento de covarianza, de otra familia que la logística. Mi primer intento (logística balanceada, carpeta `sprint_2/`) fue reemplazado: su «mejora» era un cambio de umbral con probabilidades infladas. La regla nueva exige también que el log-loss interno no empeore más de 5 %.

- Resultado en 2020-2025: **costo 109** (línea base 318, −66 %), precisión 0,675, recall 0,783, umbral 0,30.
- **No supera a la referencia** (99), pero tiene probabilidades calibradas (media prevista 0,24 frente a 0,26 real): `probabilidad_hit` se puede leer como probabilidad.

El análisis completo de ambos sprints está en `modelo_nuevo/informe_modelo_nuevo.md`.

## 6. Estado de la «definición de terminado»

| Requisito | Sprint 1 | Sprint 2 |
|---|---|---|
| `pytest -q` pasa con el modelo | Hecho (9 pruebas) | Hecho (9 pruebas, con `RUTA_MODELO=models/modelo_sprint2.pkl`) |
| La ficha reporta métricas 2020-2025 y supera la línea base | Hecho (5,82 < 10,44) | Hecho (109 < 318) |
| Prueba de humo local «servicio en orden» | Hecho | Hecho |
| Publicado en Render | **Pendiente: lo haces tú** | **Pendiente: lo haces tú** |
| `probar_servicio SU_URL` termina en «servicio en orden» | **Pendiente** (necesita tu URL) | **Pendiente** |

No pude publicar en Render: requiere tu cuenta de GitHub y de Render, que no tengo.

## 7. Qué archivo va en cada sprint

- `models/modelo.pkl` = **modelo nuevo del sprint 1** (aditivo). Es el que debe estar publicado ahora.
- `models/modelo_sprint2.pkl` = **modelo nuevo del sprint 2** (LDA). Cuando comience el sprint 2, cópialo sobre `models/modelo.pkl` y sube el cambio; la URL no cambia.
- Conservados como respaldo y comparación: `models/modelo_sprint1_referencia.pkl` y `models/modelo_sprint2_balanceada.pkl`.

```bash
cp models/modelo_sprint2.pkl models/modelo.pkl
```

> **Nota sobre `semana_09/`:** esa carpeta ya venía en el ZIP. Confirma que es tu trabajo o el que debes entregar; yo solo verifiqué que se reproduce.

## 8. Ejecutar todo en local

Requiere Python 3.11, 3.12 o 3.13 (no 3.10 ni 3.14).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # versiones exactas: el .pkl solo carga con ellas

pytest -q                            # prueba el modelo de models/modelo.pkl
python -m modelo_nuevo.analisis_sprint1_nuevo   # análisis del modelo nuevo, sprint 1
python -m modelo_nuevo.analisis_sprint2_nuevo   # análisis del modelo nuevo, sprint 2
uvicorn src.app.main:app --reload    # http://127.0.0.1:8000/docs
python -m src.probar_servicio        # en otra terminal, con el servicio arriba
```

## 9. Publicar en Render, paso a paso

1. Crea en tu GitHub un repositorio propio (Fork del original o repositorio nuevo) con este contenido, incluido `models/modelo.pkl`. Con git:
   ```bash
   git init && git add . && git commit -m "Sprint 1: modelo de regresión"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   git push -u origin main
   ```
2. Entra a <https://render.com> con tu cuenta de GitHub.
3. **New +** → **Blueprint** → elige tu repositorio → **Apply**. Render lee `render.yaml` (Python 3.11.9, plan gratuito) y tarda de 3 a 5 minutos.
4. Copia la URL asignada y compruébala:
   ```bash
   python -m src.probar_servicio https://tu-servicio.onrender.com
   ```
   Debe terminar con **«servicio en orden»**. La primera petición puede tardar cerca de un minuto si el servicio estaba dormido.
5. Entrega la URL.

## 10. Problemas frecuentes

- **Aviso de versión de scikit-learn:** reentrena con las versiones de `requirements.txt` (scikit-learn 1.6.1). Los modelos entregados ya se crearon con ellas.
- **`503` en `/modelo`:** falta `models/modelo.pkl` en el repositorio.
- **`ModuleNotFoundError: No module named 'src'`:** ejecuta desde la raíz del repositorio.
- **`422` en `/predecir`:** la entrada no cumple el formato (género desconocido, valor fuera de rango o un campo posterior al lanzamiento como `reproducciones_sem1`; esto último es lo correcto).

## 11. Qué no debes afirmar (límites honestos)

Los datos son sintéticos, hay un solo periodo de prueba y los resultados son asociaciones, no causas. Los modelos nuevos no superan a las referencias: empatan o quedan levemente por debajo. En regresión, el modelo prioriza pero no pronostica con precisión una canción. En clasificación, los costos 3 y 1 los debe confirmar quien responde por el presupuesto.
