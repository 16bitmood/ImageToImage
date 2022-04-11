import os

from pix2pix import Pix2Pix
from datasets import SideBySideDataset

def satellite_to_map_model():
    folder = 'data/satellite_to_map'
    train_dataset = SideBySideDataset(os.path.join(folder, 'train'), output='right')
    test_dataset  = SideBySideDataset(os.path.join(folder, 'val'), output='right')

    model = Pix2Pix(
        train_dataset, 
        test_dataset,
        os.path.join(folder, 'model.pth.tar'),
        os.path.join(folder, 'example_outputs'),
        num_examples=3
    )

    model.train()

def segmentation_to_facade_model():
    folder = 'data/segmentation_to_facade'
    train_dataset = SideBySideDataset(os.path.join(folder, 'train'), output='left')
    test_dataset  = SideBySideDataset(os.path.join(folder, 'test'), output='left')

    model = Pix2Pix(
        train_dataset, 
        test_dataset,
        os.path.join(folder, 'model.pth.tar'),
        os.path.join(folder, 'example_outputs'),
        num_examples=3,
    )

    model.train()


if __name__ == '__main__':
    # satellite_to_map_model()
    segmentation_to_facade_model()