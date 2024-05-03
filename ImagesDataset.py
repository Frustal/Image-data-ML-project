import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
from to_grayscale import to_grayscale
from prepare_image import prepare_image

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, width=100, height=100, dtype=None):
        if width < 100 or height < 100:
            raise ValueError("Width and height must be >= 100")

        self.width = width
        self.height = height

        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])

        # loading class names and assigning ids
        class_file = [f for f in os.listdir(image_dir) if f.endswith('.csv')][0]
        class_data = pd.read_csv(os.path.join(image_dir, class_file), sep=';', header=0)
        self.class_dict= dict(zip(class_data.iloc[:, 0], class_data.iloc[:, 1]))
        self.class_labels = sorted(class_data.iloc[:, 1].unique())
        self.class_id_dict = {class_label: i for i, class_label in enumerate(self.class_labels)}

        self.dtype = dtype

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path)

        if self.dtype:
            image = np.array(image, dtype=self.dtype)
        else:
            image = np.array(image)

        gray_image = to_grayscale(image)

        resized_image = prepare_image(gray_image, self.width, self.height)[0]

        filename = os.path.basename(image_path)
        class_name = self.class_dict[filename]
        class_id = self.class_id_dict[class_name]

        return resized_image, class_id, class_name, image_path