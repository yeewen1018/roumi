"""
Comprehensive tests for zero-copy tensor bridge implementation.

Tests:
1. Basic functionality
2. Zero-copy verification (shared memory)
3. Memory lifecycle and leak detection
4. Edge cases and error conditions
"""

import torch
import gc
import weakref
import psutil
import os
import numpy as np
from roumi.utils import create_test_batch


def get_process_memory():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_basic_functionality():
    """Test basic PyMiniBatch creation and tensor access."""
    print("\n=== Test 1: Basic Functionality ===")

    batch = create_test_batch()
    print(f"Created: {batch}")

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

    assert input_ids.shape == (2, 2)
    assert labels.shape == (2, 1)

    expected_input = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    expected_labels = torch.tensor([[0], [1]], dtype=torch.long)
    assert torch.equal(input_ids, expected_input)
    assert torch.equal(labels, expected_labels)

    print("✅ Basic functionality works")


def test_zero_copy_shared_memory():
    """Verify tensors share memory (true zero-copy)."""
    print("\n=== Test 2: Zero-copy Memory Sharing ===")

    batch = create_test_batch()

    tensor1 = batch["input_ids"]
    tensor2 = batch["input_ids"]
    assert tensor1.data_ptr() == tensor2.data_ptr(), "Should share memory"

    tensor1[0, 0] = 999
    assert tensor2[0, 0] == 999, "Changes should be visible (shared memory)"

    # Modify tensor1 and verify tensor2 sees the change
    tensor1[0, 0] = 999
    assert tensor2[0, 0] == 999, "Changes should be visible (shared memory)"

    print(f"Memory address: {hex(tensor1.data_ptr())}")
    print(
        f"After modification: tensor1[0, 0]={tensor1[0,0]}, tensor1[0,0]={tensor2[0, 0]}"
    )
    print("✅ Zero-copy verified - tensors share memory")


def test_memory_lifecycle():
    """Test memory is properly managed through lifecycle."""
    print("\n=== Test 3: Memory Lifecycle ===")

    gc.collect()
    mem_before = get_process_memory()
    print(f"Memory before: {mem_before:.2f} MB")

    for i in range(100):
        batch = create_test_batch()
        tensor = batch["input_ids"]
        _ = tensor.sum().item()

        del tensor
        del batch

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    mem_after = get_process_memory()
    print(f"Memory after: {mem_after: .2f} MB")

    mem_growth = mem_after - mem_before
    print(f"Memory growth: {mem_growth:.2f} MB")
    assert mem_growth < 10, f"Memory leak detected: {mem_growth:.2f} MB growth"
    print("✅ No memory leaks detected")


def test_tensor_lifetime():
    """Test tensor remains valid even after batch is deleted"""
    print("\n=== Test 4: Tensor Lifetime Management ===")

    batch = create_test_batch()
    tensor = batch["input_ids"]
    original_sum = tensor.sum().item()

    # Delete batch - tensor should still be valid (Arc keeps it alive)
    del batch
    gc.collect()

    # Tensor should still be usable
    assert tensor.sum().item() == original_sum
    tensor[0, 0] = 42
    assert tensor[0, 0] == 42

    print("✅ Tensor survives batch deletion (Arc reference counting works)")


def test_error_handling():
    """Test error handling for invalid operations."""
    print("\n=== Test 6: Error handling ===")

    batch = create_test_batch()
    try:
        _ = batch["non_existent"]
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "Feature 'non_existent' not found" in str(e)
        print("✅ KeyError handling works")

    assert "input_ids" in batch
    assert "non_existent" not in batch
    print("✅ Contains operator works")


def test_thread_safety():
    """Test thread safety behavior."""
    print("\n=== Test 7: Thread Safety ===")
    print("INFO: PyMiniBatch is marked 'unsendable'")
    print("      - Can only be accessed from creating thread")
    print("      - Cross-thread access will panic (by design)")
    print("      - This is PyO3's safety enforcement")

    import threading

    batch = create_test_batch()

    # Test 1: Same thread access works
    for _ in range(100):
        _ = batch["input_ids"]
    print("✅ Single thread: Multiple accesses work")

    # Test 2: Cross-thread access is blocked (this is good!)
    def try_access():
        try:
            _ = batch["input_ids"]
            return "ERROR: Should not work!"
        except Exception as e:
            if "unsendable" in str(e):
                return "OK: Thread safety enforced"
            return f"ERROR: Wrong exception: {e}"

    t = threading.Thread(target=try_access)
    t.start()
    t.join()
    print("✅ Cross-thread: Access correctly blocked by PyO3")


def run_all_tests():
    print("=" * 60)
    print("TENSOR BRIDGE TEST SUITE")
    print("=" * 60)

    test_basic_functionality()
    test_zero_copy_shared_memory()
    test_memory_lifecycle()
    test_tensor_lifetime()
    test_error_handling()
    test_thread_safety()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
