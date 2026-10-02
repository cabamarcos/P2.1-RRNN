# Clasificación de medios de transporte

[![CI](https://github.com/cabamarcos/P2.1-RRNN/actions/workflows/ci.yml/badge.svg)](https://github.com/cabamarcos/P2.1-RRNN/actions/workflows/ci.yml)

<!-- academic-catalog:start -->
**UC3M · 4.º curso · Redes de neuronas**

Redes neuronales densas para identificar cinco medios de transporte a partir de sensores. Incluye particiones por trayectoria, comparación de ponderación de clases, evaluación reproducible e inferencia con un modelo guardado.

**Tecnologías:** Python, TensorFlow, Keras, scikit-learn, pandas, Matplotlib.

[Ver todos mis proyectos académicos](https://github.com/cabamarcos/academic-projects)
<!-- academic-catalog:end -->

## El problema

Clasificar desplazamientos **andando, en bicicleta, en autobús, en coche o en metro**
con las estadísticas de sensores incluidas en `dataRRNN2.csv`. Son 29.151 muestras,
20 atributos predictivos y dos identificadores de usuario y trayectoria. Las clases
están desbalanceadas: 10.899 muestras andando frente a 1.748 en metro.

Proyecto académico de **Pablo Hidalgo Delgado y Marcos Caballero Cortés**. La entrega
original se conserva en [archive](archive/README.md). Los dos cuadernos de la raíz
se han completado y utilizan el mismo pipeline probado que la línea de comandos.

## Resultados de la ejecución incluida

| Modelo | Accuracy test | F1 macro test |
|---|---:|---:|
| Clase mayoritaria | 37,47 % | 0,1090 |
| Regresión logística ponderada | 77,34 % | 0,7361 |
| **MLP seleccionado** | **83,47 %** | **0,8045** |

La evaluación utiliza **9.799 filas de trayectorias reservadas**. No se comparte
ninguna trayectoria entre entrenamiento, validación y prueba. Se selecciona la red
por F1 macro de validación: la red sin ponderación alcanzó 0,8385 y la ponderada
0,8268. El test se consulta después de esa selección.

![Matriz de confusión del modelo en test](results/track-42/confusion_matrix.png)

[Métricas completas y configuración](results/track-42/metrics.json) ·
[Curvas de validación](results/track-42/validation_loss.png) ·
[Predicciones de test](results/track-42/test_predictions.csv) ·
[Modelo guardado](results/track-42/model.keras)

## Reproducir o usar el modelo

Entorno probado: **Python 3.12 y TensorFlow 2.20, en CPU**.

```sh
python -m venv .venv
```

Activa el entorno con `.venv\Scripts\Activate.ps1` en PowerShell o
`source .venv/bin/activate` en Linux/macOS. Después:

```sh
python -m pip install -r requirements-lock.txt
python -m transport train --data dataRRNN2.csv --output runs/track-42
```

Se entrenan dos MLP de 64 y 32 neuronas, con ReLU, salida softmax, Adam y entropía
cruzada. Se comparan pesos de clase frente a entrenamiento sin ponderación. Early
stopping restaura los mejores pesos: máximo 80 épocas, paciencia 10. La semilla es 42.
Cada experimento debe utilizar una carpeta nueva para conservar los anteriores.

Para predecir con el modelo incluido:

```sh
python -m transport predict --model results/track-42 --data examples/sensors.csv --output runs/predictions.csv
```

El CSV debe contener las 20 columnas de sensores; el orden de las columnas no importa.
La etiqueta y los identificadores no son necesarios. La salida contiene la clase y
las cinco probabilidades. Se aplica la normalización guardada, sin volver a ajustarla.

## Método y límites de la evaluación

La separación por defecto reserva un tercio de los grupos de trayectoria para test
y el 20 % de los grupos restantes para validación. En esta ejecución quedan **15.264
filas de entrenamiento, 4.088 de validación y 9.799 de prueba**. Las trayectorias se
identifican por la pareja `(user_id, track_id)`; sus longitudes explican que las
proporciones por fila varíen.

Los identificadores se excluyen de las entradas. La estandarización se ajusta solo
con train. Los pesos de clase también se calculan exclusivamente con train. Las
arquitecturas se comparan con validación y el conjunto de test queda reservado para
la evaluación final. Accuracy y F1 por clase acompañan al F1 macro para interpretar
el desbalanceo.

**Hay usuarios compartidos entre conjuntos.** Estas cifras evalúan trayectorias nuevas,
y no permiten afirmar el rendimiento sobre personas nuevas. `--split user` ofrece
una separación por usuario; `--split row` conserva una comparación con el reparto
estratificado original. No se publican resultados de esos dos modos sin ejecutarlos.
Una sola semilla tampoco estima la variabilidad del rendimiento.

## Cuadernos y comprobaciones

- [P2.1_RRNN.ipynb](P2.1_RRNN.ipynb): exploración, método, selección, evaluación e interpretación.
- [RRNN2.ipynb](RRNN2.ipynb): receta completa desde el CSV hasta las predicciones.

Los cuadernos consultan por defecto el experimento incluido. `RETRAIN = True` genera
otro entrenamiento. Ábrelos en un editor compatible con Jupyter desde la raíz del
repositorio y selecciona el entorno `.venv`; `requirements-lock.txt` incluye el
kernel y las dependencias necesarias para ejecutarlos.

```sh
python -m pytest -q
python -m ruff check transport tests scripts
python scripts/check_notebooks.py
```

**17 pruebas** comprueban particiones disjuntas, reproducibilidad, normalización sin
fuga de información, validación del CSV, entrenamiento real reducido y predicción
tras guardar y cargar el modelo. GitHub Actions también ejecuta ambos cuadernos.

El experimento guarda la configuración, las versiones, el hash del dataset, los
índices de cada partición, las curvas de entrenamiento, el escalador, el modelo y
las predicciones. Las cifras de arriba proceden de esa ejecución y no sustituyen
los resultados históricos del cuaderno archivado.
