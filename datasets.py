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
        image = Image.open(img_path)

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