from pathlib import Path
import os.path
import pytest

import torch
from torch.utils.data import Dataset

from my_project.data import MyDataset
from pathlib import Path

data_path = Path("data/processed/")

@pytest.mark.skipif(not os.path.exists(data_path), reason="Data files not found")
def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset()
    train_images, train_target, test_images, test_target = dataset.load(data_path)
    assert train_images.shape == (30000, 1, 28, 28)
    assert len(train_target) == 30000
    assert test_images.shape == (5000, 1, 28, 28)
    assert len(test_target) == 5000
    
    train_targets = torch.unique(train_target)
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test_target)
    assert (test_targets == torch.arange(0,10)).all()
    
