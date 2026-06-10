# Sistema de Recomendación de Moda basado en Visión por Computadora

Proyecto desarrollado en 9° semestre de la carrera (curso de Visión por
Computadora, Universidad del Rosario). Implementa un sistema de
recomendación **item-to-item**: dada una imagen de una prenda, el sistema
recomienda otras prendas visualmente similares, sin necesidad de metadata
como categoría, precio o marca.

## Idea general

Los sistemas de recomendación clásicos (item-to-item) suelen comparar
productos usando atributos tabulares: categoría, precio, reputación del
vendedor, ventas históricas, etc. En este proyecto reemplazamos esa
información por **una representación visual de cada producto**: convertimos
cada imagen en un vector numérico ("embedding") que captura sus
características visuales (forma, color, textura, estilo), y luego buscamos
los productos cuyos vectores estén más cerca entre sí.

## Dataset

Se usa el dataset público **[Fashion Product Images
(Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)**
de Kaggle: ~44,000 imágenes de prendas y accesorios, junto con un
`styles.csv` que incluye metadata (categoría, color, género, temporada,
nombre del producto, etc.).

> El dataset **no está incluido en este repositorio** (ver sección
> [Archivos no incluidos](#archivos-no-incluidos-y-por-qué)). Para
> reproducir el proyecto, descárgalo desde el link de Kaggle y colócalo en
> `fashion-dataset/` con esta estructura:
>
> ```
> fashion-dataset/
> ├── images/        # ~44,000 archivos .jpg
> ├── styles/
> ├── images.csv
> └── styles.csv
> ```

## Pipeline del proyecto

El flujo completo está en `recomendacion_final.ipynb`. Se divide en tres
pasos:

### 1. Vectorización de imágenes con EfficientNetB1

Cada imagen se redimensiona a 100x100 px y se pasa por **EfficientNetB1**
(red convolucional pre-entrenada en ImageNet), usando solo la parte
convolucional (sin la capa final de clasificación) más una capa de
`GlobalAveragePooling2D`. El resultado es un **vector de 1280 dimensiones**
por imagen, que resume su contenido visual sin necesidad de entrenar nada
desde cero (esto se conoce como *transfer learning*: aprovechamos lo que la
red ya "aprendió" de millones de imágenes para extraer features útiles en
nuestro dominio).

Este paso se aplica a las ~44,000 imágenes del dataset y el resultado se
guarda en `features_list.pkl` (un array de NumPy con shape `(44441, 1280)`).

### 2. Baseline: búsqueda de vecinos cercanos con KNN

Como primer acercamiento, se usa `KNeighborsClassifier` de scikit-learn
para encontrar, dado el vector de una imagen de consulta, los 25 vectores
más cercanos en el espacio de 1280 dimensiones (usando distancia
euclidiana). Esto sirve como **referencia** para comparar contra el método
aproximado del siguiente paso.

Limitación: KNN exacto recorre *todos* los vectores en cada consulta —
funciona bien con 44k imágenes, pero no escalaría a catálogos de millones de
productos.

### 3. Búsqueda aproximada con Annoy

Para resolver el problema de escalabilidad, se construye un índice con
**[Annoy](https://github.com/spotify/annoy)** (Approximate Nearest
Neighbors Oh Yeah, de Spotify), usando la métrica `dot` (producto punto).
Annoy pre-construye una estructura de árboles que permite encontrar vecinos
"suficientemente buenos" en tiempo casi constante, en vez de comparar contra
todo el dataset.

El índice construido se guarda en `image_features_dot.ann` (también se
generó una versión exploratoria con otra métrica en
`image_features.ann`).

**Resultado**: dada una imagen de consulta, el sistema devuelve las 25
imágenes más similares visualmente (mismo tipo de prenda, colores o
patrones parecidos), tanto con KNN como con Annoy, permitiendo comparar la
calidad de ambos enfoques.

## Estructura de archivos

```
vision-recomendacion/
├── recomendacion_final.ipynb   # Notebook principal: pipeline completo
├── recomendacion.ipynb         # Versión exploratoria/previa
├── Untitled.ipynb               # Notas y pruebas sueltas
├── fashion-dataset/             # Dataset (no versionado, ver abajo)
├── features_list.pkl            # Vectores de características (no versionado)
├── image_features.ann           # Índice Annoy, métrica euclidiana (no versionado)
└── image_features_dot.ann       # Índice Annoy, métrica dot (no versionado)
```

## Archivos no incluidos y por qué

Este repositorio **no incluye** los siguientes archivos/carpetas (ver
`.gitignore`):

| Archivo/carpeta | Tamaño aprox. | Por qué no se sube |
|---|---|---|
| `fashion-dataset/` | ~44,000 imágenes (varios GB) | Dataset público de terceros; demasiado pesado y son demasiados archivos para versionar en Git. Se descarga aparte desde Kaggle. |
| `features_list.pkl` | ~217 MB | Supera el límite de 100MB por archivo de GitHub. Es un **artefacto generado**: se recrea corriendo la sección de vectorización del notebook sobre el dataset. |
| `image_features.ann` | ~227 MB | Mismo caso: índice Annoy generado a partir de `features_list.pkl`. Se reconstruye en minutos con el notebook. |
| `image_features_dot.ann` | ~227 MB | Igual que el anterior, con métrica `dot`. |

En resumen: lo que se versiona es el **código** (los notebooks); los
**datos y artefactos pesados** se regeneran localmente ejecutando el
notebook con el dataset descargado.

## Cómo reproducirlo

1. Descargar el dataset de Kaggle y colocarlo en `fashion-dataset/` con la
   estructura indicada arriba.
2. Instalar dependencias: `tensorflow`, `keras`, `numpy`, `pandas`,
   `opencv-python`, `scikit-learn`, `annoy`, `matplotlib`.
3. Abrir `recomendacion_final.ipynb` y ejecutar las celdas en orden:
   - Vectorización de imágenes (genera `features_list.pkl`).
   - Baseline KNN sobre los vectores.
   - Construcción del índice Annoy (genera `image_features_dot.ann`) y
     comparación de resultados.

## Estado y próximos pasos

Proyecto migrado al portafolio para mostrar el enfoque y los resultados.
Ideas para evolucionarlo:

- Empaquetar el pipeline en funciones/scripts reutilizables (fuera del
  notebook).
- Probar otros modelos de vectorización (CLIP, ResNet, etc.) y comparar
  calidad de las recomendaciones.
- Servir el modelo detrás de una API simple para hacer demos interactivas.

## Tecnologías usadas

- **TensorFlow / Keras** — modelo pre-entrenado EfficientNetB1 (transfer
  learning).
- **scikit-learn** — KNN como baseline.
- **Annoy (Spotify)** — búsqueda aproximada de vecinos cercanos.
- **NumPy / Pandas / OpenCV / Matplotlib** — procesamiento de datos e
  imágenes.
