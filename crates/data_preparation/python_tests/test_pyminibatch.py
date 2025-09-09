import torch
import gc
import weakref
import psutil
import os
from roumi.utils import create_test_batch

def test_basic_pyminibatch_functionality():
    """Test basic PyMiniBatch creation and dict-like access."""
    print("=== Basic PyMiniBatch Functionality ===")
    
    # Create test batch
    batch = create_test_batch()
    print(f"Created: {batch}")
    print(f"Batch size: {batch.batch_size}")
    print(f"Features: {batch.features}")
    
    # Test dict-like access
    images = batch['input_ids']
    labels = batch['labels']
    
    print(f"Images tensor: shape={images.shape}, dtype={images.dtype}")
    print(f"Labels tensor: shape={labels.shape}, dtype={labels.dtype}")
    print(f"Images values: {images}")
    print(f"Labels values: {labels}")
    
    # Test that these are actual PyTorch tensors
    assert isinstance(images, torch.Tensor), "Should return PyTorch tensor"
    assert isinstance(labels, torch.Tensor), "Should return PyTorch tensor"
    
    print("✅ Basic functionality works")
    return batch

if __name__ == '__main__':
    batch = test_basic_pyminibatch_functionality()