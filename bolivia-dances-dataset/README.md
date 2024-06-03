# TODA ESTA INFO ESTA SACADA DE:

[https://www.tensorflow.org/datasets/cli?hl=es-419](https://www.tensorflow.org/datasets/add_dataset?hl=es-419)
[https://www.tensorflow.org/datasets/add_dataset?hl=es-419](https://www.tensorflow.org/datasets/add_dataset?hl=es-419)

# INSTALAR CLI DE TENSORFLOW DATASETS

```bash
!pip install -q tfds-nightly
!tfds --version
```

# PARA COMPROBAR QUE FUNCIONA CORRECTAMENTE

```bash
!tfds --help
```

# PARA CONSTRUIR SU DATASET

```bash
!tfds build mi_dataset.py
```

# AL CONSTRUIR SU DATASET, PUEDEN USARLO DE LA SIGUIENTE MANERA

```python
import tensorflow_datasets as tfds

# Cargar el dataset
ds = tfds.load('mi_dataset')
```