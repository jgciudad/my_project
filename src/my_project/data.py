from pathlib import Path
import os

import typer
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path = None) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(os.listdir(self.data_path))

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        
        train_image_files = [f for f in os.listdir(self.data_path) if f.startswith("train_images")]
    
        train_images, train_target = [], []
        for f in train_image_files:
            train_images.append(torch.load(os.path.join(self.data_path, f)))
            train_target.append(torch.load(os.path.join(self.data_path, 'train_target_' + f[-4:])))
        train_images = torch.cat(train_images)
        train_target = torch.cat(train_target)

        test_images = torch.load(os.path.join(self.data_path, 'test_images.pt'))
        test_target = torch.load(os.path.join(self.data_path, 'test_target.pt'))

        train_images = train_images.unsqueeze(1).float()
        test_images = test_images.unsqueeze(1).float()
        
        # Standardize train_images and test_images
        train_mean, train_std = train_images.mean([1, 2, 3], keepdim=True), train_images.std([1, 2, 3], keepdim=True)
        test_mean, test_std = test_images.mean([1, 2, 3], keepdim=True), test_images.std([1, 2, 3], keepdim=True)
        train_images = (train_images - train_mean) / train_std
        test_images = (test_images - test_mean) / test_std
        
        # Save the preprocessed data
        torch.save(train_images, output_folder / 'train_images.pt')
        torch.save(train_target, output_folder / 'train_target.pt')
        torch.save(test_images, output_folder / 'test_images.pt')
        torch.save(test_target, output_folder / 'test_target.pt')
        
    def load(self, folder: Path) -> torch.Tensor:
        """Load the preprocessed data from the output folder."""
        train_images = torch.load(folder / 'train_images.pt')
        train_target = torch.load(folder / 'train_target.pt')
        test_images = torch.load(folder / 'test_images.pt')
        test_target = torch.load(folder / 'test_target.pt')
        
        return train_images, train_target, test_images, test_target

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
    

