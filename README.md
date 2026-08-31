# Datasets de la cátedra — NLP & Deep Learning

Dos datasets de **clasificación de texto supervisada**, listos para usar en toda la cadena de pre-entregas (Módulos 2 → 3 → 4 → Proyecto Final). Ambos vienen limpios, balanceados o estratificados, con `train`/`test` ya separados y sin valores nulos.

> **Regla de oro:** elegí **un** dataset al inicio del **Módulo 2** y usá el **mismo** hasta el Proyecto Final. No cambies de corpus a mitad del camino.

## Los dos datasets

| | **AG News** | **BBC News** |
|---|---|---|
| Idioma | Inglés | Inglés |
| Tarea | Clasificar noticias por tema | Clasificar noticias por sección |
| Clases | 4 (`World`, `Sports`, `Business`, `Sci_Tech`) | 5 (`business`, `entertainment`, `politics`, `sport`, `tech`) |
| Train / Test | 8.000 / 2.000 | 1.700 / 425 |
| Tipo de texto | **Documentos cortos** (mediana ≈ 38 palabras) | **Artículos largos** (mediana ≈ 332 palabras) |
| Balance | Perfectamente balanceado | Estratificado (leve desbalanceo natural) |
| Ideal para enseñar | `max_len` corto, entrenamiento rápido con LoRA | percentil-95 de `max_len`, truncation, desbalanceo |

**¿Cuál recomendar?** Por defecto, **AG News**: es el más liviano y entrena rápido en la GPU gratuita de Colab (clave para el Módulo 4). **BBC News** es la mejor opción si querés que el ejercicio del percentil-95 de longitud y el análisis de desbalanceo (F1 macro vs. weighted) sean más ricos, a costa de artículos más largos.

## Formato

Cada archivo es un CSV con dos columnas:

- `text`: el texto de la noticia (ya normalizado en espacios/saltos de línea).
- `label`: la categoría (string).

```
text,label
"Red Hat Appoints New CFO. September 2, 2004 ...",Business
```

## Cómo usarlos en cada pre-entrega

- **Módulo 2 (EDA):** cargá el `train.csv`, pasá `preprocess_text` sobre `text`, y hacé el EDA (histograma de longitud, N-gramas y **distribución de `label`**).
- **Módulo 3 (TF-IDF):** `TfidfVectorizer` sobre `text`, clasificador clásico sobre `label`. Baseline de referencia ya verificado: **F1-macro ≈ 0.90 (AG News)** y **≈ 0.98 (BBC)** con LogisticRegression.
- **Módulo 4 (LoRA):** convertí `label` a enteros y pasá a `datasets` de Hugging Face (ver `cargar_dataset.py`). Modelos base sugeridos: `distilbert-base-uncased` o `bert-base-uncased` (ambos en inglés, acordes al idioma del corpus).
- **Proyecto Final:** compará el baseline TF-IDF contra el Transformer + LoRA sobre **el mismo `test.csv`**.

Ver `cargar_dataset.py` para el snippet de carga (pandas y Hugging Face).

## Requisito de entorno (Módulo 4)

El fine-tuning necesita GPU. Usá **Google Colab** o **Kaggle** (GPU gratuita). Verificá con:

```python
import torch; print(torch.cuda.is_available())
```

## Fuentes y licencia

- **AG News**: corpus de AG (Antonio Gulli), de uso académico estándar (Zhang et al., 2015). Muestra balanceada y recortada por la cátedra.
- **BBC News**: dataset de D. Greene & P. Cunningham (2006), publicado para uso no comercial en investigación. Se conservan todos los artículos, con split estratificado 80/20.

Ambos se distribuyen aquí solo con fines educativos. Si publicás resultados, citá las fuentes originales.



## Mi proyecto — Clasificación de AG News con TF-IDF

### Objetivo

El objetivo de este proyecto es desarrollar un clasificador supervisado
de noticias utilizando técnicas de Procesamiento de Lenguaje Natural (NLP)
y Machine Learning clásico.

Se utiliza el dataset AG News proporcionado por la cátedra, compuesto por
cuatro categorías: World, Sports, Business y Sci_Tech.

### Preprocesamiento

Se reutilizó el pipeline de preprocesamiento desarrollado en el Módulo 2,
incluyendo limpieza de entidades HTML, eliminación de URLs y caracteres
especiales, normalización del texto, tokenización, lematización y
eliminación de stopwords.

### Vectorización TF-IDF

Para transformar los textos en una representación numérica se utilizó
`TfidfVectorizer`.

Se experimentó con diferentes configuraciones de:

- `max_features`
- `ngram_range`

Las configuraciones evaluadas fueron:

| Configuración | max_features | ngram_range |
|---|---:|---|
| A | 5.000 | (1,1) |
| B | 10.000 | (1,2) |
| C | 20.000 | (1,2) |

La configuración C obtuvo el mejor desempeño y fue seleccionada para el
modelo final.

Es importante destacar que el vectorizador se ajustó únicamente sobre el
conjunto de entrenamiento mediante `fit_transform`, mientras que sobre el
conjunto de prueba se utilizó solamente `transform`, evitando así Data
Leakage.

### Modelo

Se seleccionó Regresión Logística como modelo baseline debido a su buen
desempeño en problemas de clasificación de texto y su eficiencia para
trabajar con matrices dispersas generadas por TF-IDF.

### Resultados

La configuración seleccionada alcanzó:

- **Accuracy:** 89,65 %
- **F1 Macro:** 0,8965

El análisis de la matriz de confusión muestra un buen desempeño general.
La categoría Sports presenta el mejor comportamiento, mientras que las
principales dificultades se observan entre Business y Sci_Tech.

## Instalación y requisitos

Instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt

### Próximos pasos

Este modelo constituye el baseline del proyecto y permitirá comparar su
desempeño con modelos de Deep Learning en las siguientes etapas.
