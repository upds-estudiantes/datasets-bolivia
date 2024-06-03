"""my_dataset dataset."""
import os
from labels import get_labels

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = tfds.core.Version('1.0.3')
    RELEASE_NOTES = {
        '1.0.3': 'Initial release.',
    }

    TRAIN_URL = "data/train"
    TEST_URL = "data/test"

    LABELS = [*get_labels(TRAIN_URL)]

    IMAGE_SHAPE = (None, None, 3)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=self.IMAGE_SHAPE),
                'label': tfds.features.ClassLabel(names=self.LABELS),
            }),
            supervised_keys=('image', 'label'),
            homepage='https://mi-dataset/danzas',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Download the data as specified in `_data_url` and write it to `downloaded_path`.
        #!path = dl_manager.download_and_extract('https://data-url')

        #ruta de los datos

        return {
            #!'train': self._generate_examples(path / 'train_imgs'),

            #generar ejemplos tipos de datos (train, test, val)
            'train': self._generate_examples(self.TRAIN_URL),
        }

    def _generate_examples(self, path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)

            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, file_name)
                    image_id = "%s_%s" % (folder, file_name)
                    yield image_id, {
                        'image': image_path,
                        'label': folder,
                    }
