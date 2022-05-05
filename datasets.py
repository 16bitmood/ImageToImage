import constants
import os
from PIL import Image
from torch.utils.data import Dataset

class SideBySideDataset(Dataset):
    def __init__(self, root_dir, output = 'right'):
        self.root_dir   = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.output = output

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = Image.open(img_path).convert('RGB')

        w,h = image.size

        img1 = image.crop((0, 0, w//2, h))
        img2 = image.crop((w//2, 0, w, h))

        img1 = constants.BASE_TRANSFORMATIONS(img1)
        img2 = constants.BASE_TRANSFORMATIONS(img2)

        if self.output == 'right':
            input_image, output_image = img1, img2
        elif self.output == 'left':
            input_image, output_image = img2, img1

        return input_image, output_image

class UnpairedDataset(Dataset):
    def __init__(self, root_dir_X, root_dir_Y):
        self.root_dir_X   = root_dir_X
        self.list_files_X = os.listdir(self.root_dir_X)

        self.root_dir_Y   = root_dir_Y
        self.list_files_Y = os.listdir(self.root_dir_Y)

    def __len__(self):
        return max(len(self.list_files_X), len(self.list_files_Y))

    def __getitem__(self, index):
        img_file_X = self.list_files_X[index % len(self.list_files_X)]
        img_path_X = os.path.join(self.root_dir_X, img_file_X)
        image_X = Image.open(img_path_X).convert('RGB')

        img_file_Y = self.list_files_Y[index % len(self.list_files_Y)]
        img_path_Y = os.path.join(self.root_dir_Y, img_file_Y)
        image_Y = Image.open(img_path_Y).convert('RGB')

        image_X = constants.BASE_TRANSFORMATIONS(image_X)
        image_Y = constants.BASE_TRANSFORMATIONS(image_Y)

        return image_X, image_Y
    