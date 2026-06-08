# Vision Recomendación — Contexto del Proyecto

## Qué es
Sistema de recomendación basado en visión por computadora sobre un dataset de moda
("fashion-dataset"). Extrae características visuales de imágenes y usa índices de
vecinos cercanos aproximados (ANN) para recomendar productos similares. Proyecto
de la carrera (9° semestre, Universidad del Rosario).

## Notebooks principales
- `recomendacion_final.ipynb` — versión final del flujo de recomendación
- `recomendacion.ipynb` — versión previa/exploratoria
- `Untitled.ipynb` — notas/pruebas sueltas

## Archivos y carpetas ignorados en Git (ver `.gitignore`)
- `features_list.pkl` — vector de características extraídas del dataset (~227MB)
- `image_features.ann` / `image_features_dot.ann` — índices de búsqueda aproximada
  de vecinos cercanos (~237MB cada uno)
- `fashion-dataset/` — dataset de imágenes de moda (~178,000 archivos). No se
  versiona por su tamaño y cantidad de archivos; vive solo en disco local.

Los tres primeros se generan al ejecutar las celdas de entrenamiento/extracción
de los notebooks. El dataset de imágenes debe obtenerse por separado (ver fuente
original del proyecto) y colocarse en esta carpeta para poder ejecutar el notebook.

## Estado
Proyecto migrado al portafolio para mostrarlo a reclutadores. Pendiente: limpiar
y documentar el código para que sea legible por terceros.
