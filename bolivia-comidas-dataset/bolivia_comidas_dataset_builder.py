"""my_dataset dataset."""
import os
import zipfile
import requests
import tensorflow_datasets as tfds
import gdown

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    TRAIN_URL = "data/train"
    DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1b8-zafV-reaZFmmAxENXya6iLAO8i6Se"
    EXTRACTED_PATH = "extracted_data"
    
    LABELS = [
                        'Aji_de_pataskha', 'caldo_de_bagre', 'Cazuelas', 'Chairo', 'Charquekan_Orureno',
                        'chicharron', 'Chorizos', 'cunape', 'empanada_de_arroz', 'Falso_Conejo', 'faropa',
                        'fricase', 'fritanga', 'Karapecho', 'keperi_beniano', 'La_kalapurka', 'locro',
                        'majao', 'Mondongo', 'pacu', 'Pejerrey', 'Picana', 'Picante_de_Pollo', 'pique',
                        'Ranga', 'Saice', 'silpancho', 'Sopa_de_Mani', 'Sopa_De_Quinua', 'sudado_de_surubi'
                    ]

    IMAGE_SHAPE = (None, None, 3)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=self.IMAGE_SHAPE),
                'label': tfds.features.ClassLabel(names=self.LABELS),
            }),
            supervised_keys=('image', 'label'),
            homepage='https://mi-dataset/comidas',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        zip_path = os.path.join(dl_manager.download_dir, 'comidas-bolivia-data.zip')
        if not os.path.exists(zip_path):
            gdown.download(self.DOWNLOAD_URL, zip_path, quiet=False)
            
        extracted_data_path = os.path.join(dl_manager.download_dir, self.EXTRACTED_PATH)

        if not os.path.exists(extracted_data_path):
            os.makedirs(extracted_data_path)
            
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_data_path)
        
        return {
            'train': self._generate_examples(extracted_data_path),
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


