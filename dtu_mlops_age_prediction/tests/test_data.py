import sys
sys.path.append('./dtu_mlops_age_prediction/data')

from pathlib import Path
import torch
import pytest


@pytest.mark.skipif(not Path("./data").exists(), reason="Data directory not found")
def test_data():
    # Check if the processed files exist
    processed_dir = Path('data/processed')
    train_data_path = processed_dir / 'train_data.pt'
    test_data_path = processed_dir / 'test_data.pt'
    train_labels_path = processed_dir / 'train_labels.pt'
    test_labels_path = processed_dir / 'test_labels.pt'

    assert train_data_path.exists(), f"{train_data_path} not found"
    assert test_data_path.exists(), f"{test_data_path} not found"
    assert train_labels_path.exists(), f"{train_labels_path} not found"
    assert test_labels_path.exists(), f"{test_labels_path} not found"

    # Load the processed data
    train_data = torch.load(train_data_path)
    test_data = torch.load(test_data_path)
    train_labels = torch.load(train_labels_path)
    test_labels = torch.load(test_labels_path)

    # Test the shape of the data
    assert len(train_data) == len(train_labels), "Number of train data samples and labels do not match"
    assert len(test_data) == len(test_labels), "Number of test data samples and labels do not match"

    if len(train_data) > 0:
        # Check the shape of the first data point
        assert train_data[0].shape == torch.Size([3, 200, 200]), "Incorrect shape for train data" #MIRAR CUAL ES LA SHAPE!! 64 * 50 * 50, 128
        assert test_data[0].shape == torch.Size([3, 200, 200]), "Incorrect shape for test data"
