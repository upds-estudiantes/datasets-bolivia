"""Instrumentos dataset."""
import os
import zipfile
import tensorflow_datasets as tfds
import gdown

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Instrumentos dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    TRAIN_URL = "data/train"
    DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1Yzxl54LrjeLxxvz8Hwb8JLDhFF1Y5m6W"
    EXTRACTED_PATH = "extracted_data"
    
    LABELS =['Bombo Wankara', 'Caja', 'Caña', 'Charango', 'Erke', 'Guitarra', 'Maracas', 'Pinquillo', 'Quena', 'Maracas', 'Tarka o Quena', 'Violin', 'Zampoña']
    
    IMAGE_SHAPE = (None, None, 3)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=self.IMAGE_SHAPE),
                'label': tfds.features.ClassLabel(names=self.LABELS),
            }),
            supervised_keys=('image', 'label'),
            homepage='https://mi-dataset/Instrumentos',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        zip_path = os.path.join(dl_manager.download_dir, 'instrumentos.zip')
        if not os.path.exists(zip_path):
            gdown.download(self.DOWNLOAD_URL, zip_path, quiet=False)
            
        extracted_data_path = os.path.join(dl_manager.download_dir, self.EXTRACTED_PATH)

        if not os.path.exists(extracted_data_path):
            os.makedirs(extracted_data_path)
            
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_data_path)
        
        fullPath = os.path.join(extracted_data_path, "Redimencion y rotacion")
        
        return {
            'train': self._generate_examples(fullPath ),
        }

    def _generate_examples(self, path):
        if os.path.isdir(path):
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
        else:
            raise ValueError(f"Expected a directory at {path}, but found a file. Please check the extraction process.")
