# benches/imagenet/pytorch_cpu_gpu.py

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision
import time
import numpy as np
from pathlib import Path
import json
import psutil
import argparse
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import platform
from collections import defaultdict
import os
import gc
import threading
from contextlib import contextmanager

# GPU-specific imports (optional)
try:
    import pynvml

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring will be limited.")


@dataclass
class ComponentBenchmarkResult:
    """Results for component-level benchmarks"""

    component: str
    operation: str
    throughput: float
    unit: str
    avg_time_ms: float
    std_time_ms: float
    memory_mb: float


@dataclass
class TransformBenchmarkResult:
    """Results for transform pipeline benchmarks"""

    transform_level: str
    batch_size: int
    num_workers: int

    # Performance metrics
    images_per_sec: float
    batches_per_sec: float
    avg_batch_ms: float
    p50_batch_ms: float
    p95_batch_ms: float
    p99_batch_ms: float
    std_batch_ms: float

    # Resource metrics
    cpu_percent: float
    memory_mb: float
    memory_delta_mb: float

    # Scaling efficiency
    thread_scaling_efficiency: Optional[float] = None


@dataclass
class GPUBenchmarkResult:
    """GPU-specific benchmark results with enhanced metrics"""

    transform_level: str
    batch_size: int
    num_workers: int

    # GPU Performance metrics
    gpu_util_percent: float
    gpu_memory_mb: float
    gpu_memory_peak_mb: float

    # Timing breakdown (using median for robustness)
    dataload_time_ms: float
    transfer_time_ms: float
    compute_time_ms: float
    total_time_ms: float

    # Efficiency metrics
    gpu_efficiency: float  # compute_time / total_time
    pipeline_efficiency: float  # gpu_util * gpu_efficiency

    # Transfer metrics
    transfer_bandwidth_gbps: float
    pin_memory: bool

    # Detailed memory breakdown - with defaults
    model_memory_mb: float = 0.0
    memory_overhead_mb: float = 0.0

    # Timing variability - with default
    timing_variability: Dict[str, float] = field(default_factory=dict)

    # Additional metrics - with defaults
    effective_bandwidth_gbps: float = 0.0

    # Hardware state during benchmark - with defaults
    gpu_temperature_c: float = 0.0
    gpu_power_w: float = 0.0
    gpu_clock_mhz: int = 0

    # Statistical confidence - with defaults
    num_samples: int = 0
    samples_after_outlier_removal: int = 0


class GPUMonitor:
    """Monitor GPU utilization and memory with enhanced tracking and better error handling"""

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.samples = []
        self.monitoring = False
        self.thread = None
        self.available = False
        self.error_msg = ""

        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.error_msg = "CUDA not available"
            return

        # Check if pynvml is available
        if not GPU_AVAILABLE:
            self.error_msg = (
                "pynvml not available - install with: pip install nvidia-ml-py"
            )
            return

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # Test that we can actually query the device
            test_util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            test_mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            self.available = True
            print(f"  ✓ GPU monitoring initialized (test util: {test_util.gpu}%)")

        except pynvml.NVMLError as e:
            self.error_msg = f"NVML Error: {e}"
        except Exception as e:
            self.error_msg = f"GPU monitoring init failed: {e}"

    def start_monitoring(self):
        """Start background monitoring thread"""
        if not self.available:
            print(f"  ⚠ GPU monitoring not available: {self.error_msg}")
            return

        self.monitoring = True
        self.samples = []
        print("  ✓ Started GPU monitoring")

        def monitor_loop():
            while self.monitoring:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

                    # Optional: power monitoring (some GPUs don't support this)
                    try:
                        power = (
                            pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
                        )  # Watts
                    except:
                        power = 0

                    self.samples.append(
                        {
                            "time": time.perf_counter(),
                            "gpu_util": util.gpu,
                            "memory_mb": mem.used / 1024**2,
                            "power_watts": power,
                        }
                    )
                    time.sleep(0.01)  # 100 Hz sampling
                except Exception as e:
                    print(f"  ⚠ GPU monitoring error: {e}")
                    break

        self.thread = threading.Thread(target=monitor_loop, daemon=True)
        self.thread.start()

    def stop_monitoring(self):
        """Stop monitoring and return stats"""
        if not self.available:
            return 0, 0, 0

        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)  # Don't wait forever

        if not self.samples:
            print("  ⚠ No GPU monitoring samples collected")
            return 0, 0, 0

        gpu_utils = [s["gpu_util"] for s in self.samples]
        memory_mbs = [s["memory_mb"] for s in self.samples]

        avg_util = np.mean(gpu_utils)
        avg_mem = np.mean(memory_mbs)
        peak_mem = np.max(memory_mbs)

        print(
            f"  ✓ GPU monitoring: {len(self.samples)} samples, "
            f"avg util: {avg_util:.1f}%, peak mem: {peak_mem:.1f}MB"
        )

        return avg_util, avg_mem, peak_mem

    def get_detailed_memory_stats(self):
        """Get detailed memory statistics"""
        if not self.available:
            return {
                "allocated_mb": 0,
                "reserved_mb": 0,
                "peak_allocated_mb": 0,
                "peak_reserved_mb": 0,
            }

        torch.cuda.synchronize()

        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device_id) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(self.device_id) / 1024**2,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device_id)
            / 1024**2,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device_id)
            / 1024**2,
        }


class FixedSizeImageFolder(ImageFolder):
    """ImageFolder that ensures consistent tensor sizes"""

    def __init__(self, root: str, transform=None, target_size=(320, 320)):
        super().__init__(root, transform)
        self.target_size = target_size

        # Pre-transform to ensure consistent sizes
        self.base_transform = transforms.Compose(
            [transforms.Resize(self.target_size), transforms.ToTensor()]
        )

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)

        # Apply transform if provided, otherwise use base
        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample = self.base_transform(sample)

        return sample, target


class ComprehensiveBenchmark:
    """Comprehensive CPU-GPU benchmark with component isolation and enhanced metrics"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.platform = platform.system()

        # Collect sample paths for component benchmarks
        dataset = ImageFolder(self.data_root)
        self.sample_paths = [s[0] for s in dataset.samples[:1000]]  # Use first 1000

        # Define transform levels
        self.transform_levels = self._get_transform_levels()

        # Setup GPU components if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # Simple CNN that matches ImageNet preprocessing
            self.model = torchvision.models.resnet50(pretrained=False).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            print(f"GPU available: {torch.cuda.get_device_name()}")
        else:
            self.device = None
            self.model = None
            print("No GPU available - CPU only benchmarks")

    def _get_transform_levels(self) -> Dict[str, Any]:
        """Get all transform configurations"""
        return {
            # Component isolation levels
            "level0_io_only": None,  # Special handling
            "level1_decode_only": None,  # Special handling
            "level2_tensor_only": transforms.Compose(
                [
                    transforms.Resize((224, 224)),  # Fixed size
                    transforms.ToTensor(),
                ]
            ),
            # Progressive complexity levels
            "level3_resize": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
            "level4_augment_basic": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            "level5_augment_full": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    transforms.ToTensor(),
                ]
            ),
            "level6_normalize": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "level7_production_inference": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "level8_production_training": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    transforms.RandomRotation(degrees=15),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        }

    def get_gpu_hardware_state(self):
        """Get detailed GPU hardware state for reproducibility"""
        if not torch.cuda.is_available() or not GPU_AVAILABLE:
            return {}

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Get current hardware state
            hardware_state = {
                "gpu_name": torch.cuda.get_device_name(),
                "driver_version": pynvml.nvmlSystemGetDriverVersion().decode("utf-8"),
                "cuda_version": torch.version.cuda,
                "gpu_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS
                ),
                "memory_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                ),
                "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                / 1000,
                "current_power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                "temperature_c": pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                ),
                "pcie_link_gen": pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle),
                "pcie_link_width": pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle),
                "compute_mode": pynvml.nvmlDeviceGetComputeMode(handle),
                "persistence_mode": pynvml.nvmlDeviceGetPersistenceMode(handle),
            }

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            hardware_state["total_memory_gb"] = mem_info.total / 1024**3
            hardware_state["free_memory_gb"] = mem_info.free / 1024**3

            # Get ECC state if available
            try:
                ecc_mode = pynvml.nvmlDeviceGetEccMode(handle)
                hardware_state["ecc_enabled"] = (
                    ecc_mode.current == pynvml.NVML_FEATURE_ENABLED
                )
            except:
                hardware_state["ecc_enabled"] = None

            return hardware_state
        except:
            return {}

    def clear_caches(self):
        """Clear various caches for consistent benchmarking"""
        # Clear Python's caches
        gc.collect()

        # Clear PyTorch's CUDA caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Clear OS page cache (Linux only, requires sudo)
        if self.platform == "Linux":
            try:
                # First sync to ensure all writes are flushed
                os.system("sync")

                # Try to clear caches if we have permissions
                result = os.system(
                    "sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null"
                )
                if result == 0:
                    print("  ✓ OS cache cleared")
                    time.sleep(0.5)  # Give system time to clear
                else:
                    print("  ⚠ Cannot clear OS cache (requires sudo)")
            except:
                pass

        # Clear PIL's image cache
        if hasattr(Image, "_clear_cache"):
            Image._clear_cache()

    def remove_outliers(self, times, n_std=3):
        """Remove outliers using standard deviation method"""
        times = np.array(times)
        mean = np.mean(times)
        std = np.std(times)

        # Keep values within n_std standard deviations
        mask = np.abs(times - mean) < n_std * std
        cleaned_times = times[mask]

        if len(cleaned_times) < len(times) * 0.9:  # Lost more than 10% of data
            print(
                f"  ⚠ Warning: Removed {len(times) - len(cleaned_times)} outliers "
                f"({(1 - len(cleaned_times)/len(times))*100:.1f}%)"
            )

        return cleaned_times

    def calculate_robust_stats(self, times):
        """Calculate robust statistics"""
        times = self.remove_outliers(times)

        return {
            "mean": np.mean(times),
            "median": np.median(times),
            "std": np.std(times),
            "p25": np.percentile(times, 25),
            "p50": np.percentile(times, 50),
            "p75": np.percentile(times, 75),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99),
            "iqr": np.percentile(times, 75) - np.percentile(times, 25),
            "cv": (
                np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
            ),  # Coefficient of variation
        }

    def ensure_gpu_stability(
        self, model, batch_size, target_variance=0.05, max_warmup_iters=50
    ):
        """Ensure GPU is warmed up and timing is stable"""
        if self.device is None:
            return

        print("  Ensuring GPU stability...")

        # Create dummy data
        dummy_data = torch.randn(batch_size, 3, 224, 224, device=self.device)
        dummy_target = torch.randint(0, 1000, (batch_size,), device=self.device)

        # Run warmup iterations until timing stabilizes
        recent_times = []
        window_size = 10

        for i in range(max_warmup_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()

            self.optimizer.zero_grad()
            output = model(dummy_data)
            loss = self.criterion(output, dummy_target)
            loss.backward()
            self.optimizer.step()

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            recent_times.append(elapsed)
            if len(recent_times) > window_size:
                recent_times.pop(0)

            # Check if timing has stabilized
            if len(recent_times) == window_size:
                cv = np.std(recent_times) / np.mean(recent_times)
                if cv < target_variance:
                    print(f"  ✓ GPU stabilized after {i+1} iterations (CV={cv:.3f})")
                    break
        else:
            cv = np.std(recent_times) / np.mean(recent_times)
            print(f"  ⚠ GPU timing did not stabilize (CV={cv:.3f})")

        # Clear memory
        del dummy_data, dummy_target
        torch.cuda.empty_cache()

    def benchmark_components(self) -> List[ComponentBenchmarkResult]:
        """Benchmark individual components in isolation"""
        results = []

        print("\n" + "=" * 80)
        print("COMPONENT ISOLATION BENCHMARKS")
        print("=" * 80)

        # 1. Pure I/O
        print("\n1. Pure I/O (disk read only)")
        result = self._benchmark_io_only()
        results.append(result)
        print(
            f"   → {result.throughput:.1f} {result.unit}, {result.avg_time_ms:.2f}ms avg"
        )

        # 2. I/O + Decode
        print("\n2. I/O + JPEG Decode")
        result = self._benchmark_decode_only()
        results.append(result)
        print(
            f"   → {result.throughput:.1f} {result.unit}, {result.avg_time_ms:.2f}ms avg"
        )

        # 3. Individual transform operations
        print("\n3. Transform Operations")
        for op_result in self._benchmark_transform_operations():
            results.append(op_result)
            print(
                f"   {op_result.operation}: {op_result.throughput:.1f} {op_result.unit}, "
                f"{op_result.avg_time_ms:.2f}ms avg"
            )

        return results

    def _benchmark_io_only(self) -> ComponentBenchmarkResult:
        """Measure pure disk I/O"""
        times = []
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024

        # Warmup
        for path in self.sample_paths[:10]:
            with open(path, "rb") as f:
                _ = f.read()

        # Benchmark
        for path in self.sample_paths[:500]:
            start = time.perf_counter()
            with open(path, "rb") as f:
                data = f.read()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times = np.array(times)
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024

        # Calculate file sizes for MB/s
        sizes = [Path(p).stat().st_size for p in self.sample_paths[:500]]
        avg_size_mb = np.mean(sizes) / (1024 * 1024)
        throughput_mbps = avg_size_mb / times.mean()

        return ComponentBenchmarkResult(
            component="io",
            operation="disk_read",
            throughput=throughput_mbps,
            unit="MB/s",
            avg_time_ms=times.mean() * 1000,
            std_time_ms=times.std() * 1000,
            memory_mb=mem_after - mem_before,
        )

    def _benchmark_decode_only(self) -> ComponentBenchmarkResult:
        """Measure I/O + JPEG decode"""
        times = []
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024

        # Warmup
        for path in self.sample_paths[:10]:
            _ = Image.open(path).convert("RGB")

        # Benchmark
        for path in self.sample_paths[:500]:
            start = time.perf_counter()
            img = Image.open(path).convert("RGB")
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times = np.array(times)
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024

        return ComponentBenchmarkResult(
            component="decode",
            operation="image_decode",
            throughput=1.0 / times.mean(),
            unit="images/s",
            avg_time_ms=times.mean() * 1000,
            std_time_ms=times.std() * 1000,
            memory_mb=mem_after - mem_before,
        )

    def _benchmark_transform_operations(self) -> List[ComponentBenchmarkResult]:
        """Benchmark individual transform operations"""
        results = []

        # Load test images
        test_images = []
        for path in self.sample_paths[:100]:
            img = Image.open(path).convert("RGB")
            test_images.append(img)

        # Define operations to benchmark - UPDATED to match Rust
        operations = [
            # Multiple resize sizes to match Rust
            ("resize_128x128", transforms.Resize((128, 128))),
            ("resize_256x256", transforms.Resize((256, 256))),
            ("resize_512x512", transforms.Resize((512, 512))),
            # Crop operations
            ("center_crop_224", transforms.CenterCrop(224)),
            ("random_crop_224", transforms.RandomCrop(224, pad_if_needed=True)),
            (
                "random_resized_crop_224",
                transforms.RandomResizedCrop(
                    224, scale=(0.08, 1.0), ratio=(0.75, 1.333)
                ),
            ),
            # Flip operations
            ("random_horizontal_flip", transforms.RandomHorizontalFlip(p=1.0)),
            # Rotation operations - ADDED to match Rust
            ("random_rotation_15deg", transforms.RandomRotation(degrees=15)),
            # Color operations
            (
                "color_jitter_light",
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
            ),
            (
                "color_jitter",
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
            ),
            # Tensor conversion
            ("to_tensor", transforms.ToTensor()),
            # Normalization
            (
                "normalize",
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ),
        ]

        # Prepare test data for different operations
        # Resized images for center crop
        resized_256_images = [transforms.Resize(256)(img) for img in test_images]

        # Test tensors for normalize
        test_tensors = [
            transforms.ToTensor()(transforms.Resize((224, 224))(img))
            for img in test_images
        ]

        for op_name, transform in operations:
            times = []

            # Choose appropriate input based on operation
            if op_name == "normalize":
                test_data = test_tensors
            elif op_name == "center_crop_224":
                test_data = resized_256_images
            else:
                test_data = test_images

            # Benchmark
            for data in test_data:
                start = time.perf_counter()
                try:
                    _ = transform(data)
                except Exception as e:
                    continue  # Skip if transform not applicable
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            if times:
                times = np.array(times)
                results.append(
                    ComponentBenchmarkResult(
                        component="transform",
                        operation=op_name,
                        throughput=1.0 / times.mean(),
                        unit="ops/s",
                        avg_time_ms=times.mean() * 1000,
                        std_time_ms=times.std() * 1000,
                        memory_mb=0,  # Negligible for single ops
                    )
                )

        return results

    def benchmark_transform_pipeline(
        self,
        transform_level: str,
        batch_size: int,
        num_workers: int,
        num_batches: int = 300,  # UPDATED: matches Rust
        warmup_batches: int = 30,  # UPDATED: matches Rust
        single_threaded_baseline: Optional[float] = None,
    ) -> Optional[TransformBenchmarkResult]:
        """Benchmark a complete transform pipeline"""

        print(
            f"\nBenchmarking: {transform_level}, batch_size={batch_size}, workers={num_workers}"
        )

        # Special handling for IO and decode only
        if (
            transform_level == "level0_io_only"
            or transform_level == "level1_decode_only"
        ):
            return None

        # Get transform
        transform = self.transform_levels.get(transform_level)
        if transform is None:
            print(f"  ✗ Unknown transform level: {transform_level}")
            return None

        # Create dataset with fixed output size
        dataset = FixedSizeImageFolder(
            root=self.data_root,
            transform=transform,
            target_size=(320, 320),  # Ensure consistent base size
        )

        print(f"Dataset: {len(dataset)} images")

        # Create loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,  # CPU benchmark only
            persistent_workers=num_workers > 0,  # Match Rust config
            prefetch_factor=2 if num_workers > 0 else None,  # Match Rust config
            drop_last=True,
        )

        # Memory baseline
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024

        # Measure startup time for workers
        startup_time = 0
        if num_workers > 0:
            startup_start = time.perf_counter()
            _ = next(iter(loader))
            startup_time = time.perf_counter() - startup_start
            print(f"  Worker startup time: {startup_time:.2f}s")

        # Warmup
        print(f"  Warming up with {warmup_batches} batches...")
        for i, _ in enumerate(loader):
            if i >= warmup_batches:
                break

        # Benchmark
        print(f"  Benchmarking {num_batches} batches...")
        batch_times = []
        cpu_percentages = []

        # Create iterator and time individual batches
        loader_iter = iter(loader)
        epoch_start = time.perf_counter()

        for i in range(num_batches):
            # Time individual batch fetch
            batch_start = time.perf_counter()
            try:
                images, labels = next(loader_iter)
            except StopIteration:
                print(f"  Dataset exhausted at batch {i}")
                break
            batch_end = time.perf_counter()

            batch_time = batch_end - batch_start
            batch_times.append(batch_time)

            # Ensure data is ready
            assert images.shape[0] == batch_size

            # Sample CPU
            if i % 10 == 0:
                cpu_percentages.append(psutil.cpu_percent(interval=0.01))

            # Progress
            if i % 50 == 0 or i < 10:
                elapsed = time.perf_counter() - epoch_start
                throughput = ((i + 1) * batch_size) / elapsed
                print(
                    f"    Batch {i}/{num_batches}: {batch_time*1000:.1f}ms, "
                    f"throughput: {throughput:.1f} img/s"
                )

        total_time = time.perf_counter() - epoch_start

        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024

        # Calculate thread scaling efficiency
        thread_scaling_efficiency = None
        if single_threaded_baseline is not None and num_workers > 0:
            actual_speedup = single_threaded_baseline / (total_time / len(batch_times))
            thread_scaling_efficiency = actual_speedup / num_workers

        # Calculate statistics
        batch_times = np.array(batch_times)
        actual_batches = len(batch_times)
        total_images = actual_batches * batch_size

        result = TransformBenchmarkResult(
            transform_level=transform_level,
            batch_size=batch_size,
            num_workers=num_workers,
            images_per_sec=total_images / total_time,
            batches_per_sec=actual_batches / total_time,
            avg_batch_ms=batch_times.mean() * 1000,
            p50_batch_ms=np.percentile(batch_times, 50) * 1000,
            p95_batch_ms=np.percentile(batch_times, 95) * 1000,
            p99_batch_ms=np.percentile(batch_times, 99) * 1000,
            std_batch_ms=batch_times.std() * 1000,
            cpu_percent=np.mean(cpu_percentages) if cpu_percentages else 0,
            memory_mb=mem_after,
            memory_delta_mb=mem_after - mem_before,
            thread_scaling_efficiency=thread_scaling_efficiency,
        )

        print(f"  → {result.images_per_sec:.1f} img/s, {result.avg_batch_ms:.1f}ms avg")
        if thread_scaling_efficiency is not None:
            print(
                f"  → Thread scaling efficiency: {thread_scaling_efficiency*100:.1f}%"
            )

        return result

    def benchmark_gpu_pipeline(
        self,
        transform_level: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        num_batches: int = 100,
        warmup_batches: int = 10,
    ) -> Optional[GPUBenchmarkResult]:
        """Benchmark complete pipeline with GPU - Enhanced version with better validation"""

        if self.device is None:
            print("No GPU available")
            return None

        print(
            f"\nGPU Benchmark: {transform_level}, batch={batch_size}, "
            f"workers={num_workers}, pin_memory={pin_memory}"
        )

        # Validate GPU is actually working
        print("  Validating GPU...")
        torch.cuda.synchronize()
        dummy = torch.randn(10, 3, 224, 224, device=self.device)
        dummy_out = self.model(dummy)
        torch.cuda.synchronize()
        print(f"  ✓ GPU validation passed (output shape: {dummy_out.shape})")
        del dummy, dummy_out

        # Clear all caches for consistent results
        self.clear_caches()

        # Get transform
        transform = self.transform_levels.get(transform_level)
        if transform is None:
            return None

        # Create dataset
        dataset = FixedSizeImageFolder(
            root=self.data_root, transform=transform, target_size=(256, 256)
        )

        # Create loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
        )

        # Clear GPU memory and reset stats
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Get memory after model load
        gpu_monitor = GPUMonitor()
        initial_memory = gpu_monitor.get_detailed_memory_stats()

        # Ensure GPU is warmed up and stable
        self.ensure_gpu_stability(self.model, batch_size)

        # Start GPU monitoring
        gpu_monitor.start_monitoring()

        # Storage for detailed timing
        dataload_times = []
        transfer_times = []
        compute_times = []
        total_times = []
        hardware_samples = []

        # Warmup
        print(f"  Warming up with {warmup_batches} batches...")
        for i, (images, labels) in enumerate(loader):
            if i >= warmup_batches:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            torch.cuda.synchronize()

        # Benchmark
        print(f"  Benchmarking {num_batches} batches...")

        torch.cuda.synchronize()
        total_start = time.perf_counter()

        for i, batch_data in enumerate(loader):
            if i >= num_batches:
                break

            # FIXED: More accurate timing with events
            batch_start_event = torch.cuda.Event(enable_timing=True)
            transfer_end_event = torch.cuda.Event(enable_timing=True)
            compute_end_event = torch.cuda.Event(enable_timing=True)

            # 1. Measure data loading time (CPU)
            t0 = time.perf_counter()
            images, labels = batch_data
            t1 = time.perf_counter()
            dataload_times.append(t1 - t0)

            # 2. Measure transfer time with CUDA events for accuracy
            batch_start_event.record()
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            transfer_end_event.record()

            # 3. Measure compute time
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            compute_end_event.record()

            # Wait for all operations to complete
            torch.cuda.synchronize()

            # Get GPU timing (more accurate than CPU timing for GPU ops)
            transfer_time = (
                batch_start_event.elapsed_time(transfer_end_event) / 1000
            )  # Convert to seconds
            compute_time = transfer_end_event.elapsed_time(compute_end_event) / 1000
            total_time = t1 - t0 + transfer_time + compute_time

            transfer_times.append(transfer_time)
            compute_times.append(compute_time)
            total_times.append(total_time)

            # Sample hardware state periodically
            if i % 10 == 0 and GPU_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    hardware_samples.append(
                        {
                            "temperature": pynvml.nvmlDeviceGetTemperature(
                                handle, pynvml.NVML_TEMPERATURE_GPU
                            ),
                            "power": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                            "gpu_clock": pynvml.nvmlDeviceGetClockInfo(
                                handle, pynvml.NVML_CLOCK_GRAPHICS
                            ),
                        }
                    )
                except:
                    pass

            # Progress
            if i % 20 == 0:
                total_ms = total_time * 1000
                print(
                    f"    Batch {i}/{num_batches}: {total_ms:.1f}ms total "
                    f"(load: {(t1-t0)*1000:.1f}ms, "
                    f"transfer: {transfer_time*1000:.1f}ms, "
                    f"compute: {compute_time*1000:.1f}ms)"
                )

        torch.cuda.synchronize()
        total_wall_time = time.perf_counter() - total_start

        # Stop monitoring
        avg_util, avg_mem, peak_mem = gpu_monitor.stop_monitoring()

        # Get final memory stats
        final_memory = gpu_monitor.get_detailed_memory_stats()

        # Calculate robust statistics
        dataload_stats = self.calculate_robust_stats(np.array(dataload_times) * 1000)
        transfer_stats = self.calculate_robust_stats(np.array(transfer_times) * 1000)
        compute_stats = self.calculate_robust_stats(np.array(compute_times) * 1000)

        # Calculate BETTER efficiency metrics
        # 1. Compute efficiency: how much time is spent on actual computation vs overhead
        total_pipeline_time = (
            dataload_stats["median"]
            + transfer_stats["median"]
            + compute_stats["median"]
        )
        compute_efficiency = (
            compute_stats["median"] / total_pipeline_time
            if total_pipeline_time > 0
            else 0
        )

        # 2. GPU efficiency: how well we're utilizing the GPU during compute time
        gpu_efficiency = avg_util / 100.0 if avg_util > 0 else 0

        # 3. Overall pipeline efficiency: combines both metrics
        pipeline_efficiency = compute_efficiency * gpu_efficiency

        # Calculate transfer bandwidth more accurately
        # Account for the actual tensor shapes we're transferring
        bytes_per_image = 3 * 224 * 224 * 4  # RGB float32 after transforms
        bytes_per_label = 8  # int64
        bytes_per_batch = (bytes_per_image * batch_size) + (
            bytes_per_label * batch_size
        )

        transfer_bandwidth_gbps = (bytes_per_batch / 1e9) / (
            transfer_stats["median"] / 1000
        )
        effective_bandwidth_gbps = (
            bytes_per_batch * len(transfer_times) / 1e9
        ) / np.sum(transfer_times)

        # Average hardware state
        avg_temp = (
            np.mean([s["temperature"] for s in hardware_samples])
            if hardware_samples
            else 0
        )
        avg_power = (
            np.mean([s["power"] for s in hardware_samples]) if hardware_samples else 0
        )
        avg_clock = (
            np.mean([s["gpu_clock"] for s in hardware_samples])
            if hardware_samples
            else 0
        )

        # Print validation info
        print(f"  ✓ Validation:")
        print(
            f"    - GPU Utilization: {avg_util:.1f}% (should be >0% if GPU monitoring works)"
        )
        print(
            f"    - Compute Efficiency: {compute_efficiency:.2f} (compute_time / total_time)"
        )
        print(
            f"    - Pipeline Efficiency: {pipeline_efficiency:.2f} (overall efficiency)"
        )
        print(f"    - Transfer Bandwidth: {transfer_bandwidth_gbps:.1f} GB/s")

        # Clear memory
        torch.cuda.empty_cache()

        return GPUBenchmarkResult(
            transform_level=transform_level,
            batch_size=batch_size,
            num_workers=num_workers,
            gpu_util_percent=avg_util,
            gpu_memory_mb=avg_mem,
            gpu_memory_peak_mb=peak_mem,
            model_memory_mb=initial_memory["allocated_mb"],
            memory_overhead_mb=final_memory["peak_allocated_mb"]
            - initial_memory["allocated_mb"],
            dataload_time_ms=dataload_stats["median"],
            transfer_time_ms=transfer_stats["median"],
            compute_time_ms=compute_stats["median"],
            total_time_ms=dataload_stats["median"]
            + transfer_stats["median"]
            + compute_stats["median"],
            timing_variability={
                "dataload_cv": dataload_stats["cv"],
                "transfer_cv": transfer_stats["cv"],
                "compute_cv": compute_stats["cv"],
            },
            gpu_efficiency=compute_efficiency,  # FIXED: Now represents compute/total ratio
            pipeline_efficiency=pipeline_efficiency,  # FIXED: Combined efficiency metric
            transfer_bandwidth_gbps=transfer_bandwidth_gbps,
            effective_bandwidth_gbps=effective_bandwidth_gbps,
            pin_memory=pin_memory,
            gpu_temperature_c=avg_temp,
            gpu_power_w=avg_power,
            gpu_clock_mhz=int(avg_clock),
            num_samples=num_batches,
            samples_after_outlier_removal=len(self.remove_outliers(dataload_times)),
        )

    @contextmanager
    def pytorch_profiler(self, profile_name: str):
        """Context manager for PyTorch profiling"""
        try:
            from torch.profiler import profile, ProfilerActivity, schedule

            def trace_handler(prof):
                # Save to file
                prof.export_chrome_trace(f"{profile_name}_trace.json")

                # Print summary
                print(
                    prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
                )

            with profile(
                activities=(
                    [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                    if torch.cuda.is_available()
                    else [ProfilerActivity.CPU]
                ),
                schedule=schedule(wait=2, warmup=3, active=5, repeat=1),
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                yield prof
        except ImportError:
            print("PyTorch profiler not available, skipping profiling")
            yield None

    def benchmark_with_profiler(
        self, transform_level: str, batch_size: int, num_workers: int
    ):
        """Run benchmark with detailed PyTorch profiler"""

        if not torch.cuda.is_available():
            print("GPU not available for profiling")
            return

        print(f"\nRunning profiled benchmark: {transform_level}")

        transform = self.transform_levels.get(transform_level)
        dataset = FixedSizeImageFolder(self.data_root, transform)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        with self.pytorch_profiler(
            f"profile_{transform_level}_b{batch_size}_w{num_workers}"
        ) as prof:
            if prof is not None:
                for i, (images, labels) in enumerate(loader):
                    if i >= 10:  # Profile 10 steps
                        break

                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    with torch.no_grad():
                        output = self.model(images)

                    # Step profiler
                    prof.step()

                print(f"Profile saved to profile_{transform_level}_*.json")

    def run_full_benchmark(
        self, output_file: str = "comprehensive_benchmark_results.json"
    ):
        """Run complete CPU-GPU benchmark suite with validation"""

        # Run GPU diagnostics first if GPU available
        if torch.cuda.is_available():
            print("Running GPU diagnostics...")
            if not self.run_gpu_diagnostics():
                print("GPU diagnostics failed - continuing with CPU-only benchmarks")

        # Get system info
        total_memory_gb = psutil.virtual_memory().total / (1024**3)

        results = {
            "metadata": {
                "phase": "comprehensive_cpu_gpu_benchmark",
                "data_root": str(self.data_root),
                "torch_version": torch.__version__,
                "platform": self.platform,
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "total_memory_gb": total_memory_gb,
                "has_gpu": torch.cuda.is_available(),
                "gpu_name": (
                    torch.cuda.get_device_name() if torch.cuda.is_available() else None
                ),
                "hardware_state": self.get_gpu_hardware_state(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hostname": platform.node(),
                "pynvml_available": GPU_AVAILABLE,
            },
            "component_results": [],
            "pipeline_results": [],
            "gpu_results": [],
        }

        # Platform warning
        if self.platform == "Darwin":
            print(
                "\n⚠️  Note: On macOS, PyTorch multiprocessing has significant overhead."
            )
            print(
                "   Workers may be slower for small datasets. This is a PyTorch/macOS issue.\n"
            )

        # Run component benchmarks
        print("\nPhase 1: Component Isolation")
        component_results = self.benchmark_components()
        results["component_results"] = [asdict(r) for r in component_results]

        # Run CPU pipeline benchmarks
        print("\nPhase 2: CPU Transform Pipelines")

        # Test configurations
        configs = [
            (32, 0),  # Single-threaded baseline
            (32, 4),  # Multi-threaded
            (64, 4),  # Larger batch
        ]

        for level_name in self.transform_levels.keys():
            if level_name.startswith("level0") or level_name.startswith("level1"):
                continue  # Skip, handled in components

            print(f"\n{'='*60}")
            print(f"Transform Level: {level_name}")
            print(f"{'='*60}")

            single_threaded_time = None

            for batch_size, num_workers in configs:
                try:
                    result = self.benchmark_transform_pipeline(
                        transform_level=level_name,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        single_threaded_baseline=single_threaded_time,
                    )

                    if result:
                        # Store single-threaded baseline for scaling efficiency
                        if num_workers == 0:
                            single_threaded_time = result.avg_batch_ms / 1000.0

                        results["pipeline_results"].append(asdict(result))

                except Exception as e:
                    print(f"  ✗ Error: {e}")

        # GPU Benchmarks
        if torch.cuda.is_available():
            print("\nPhase 3: GPU Pipeline Benchmarks")
            print("=" * 80)

            gpu_configs = [
                # (batch_size, num_workers, pin_memory)
                (32, 0, True),  # Single-threaded, pinned
                (32, 4, False),  # Multi-worker, no pin
                (32, 4, True),  # Multi-worker, pinned
                (64, 4, True),  # Common for V100
                (128, 4, True),  # Max batch (adjust based on GPU memory)
            ]

            for level_name in [
                "level7_production_inference",
                "level8_production_training",
            ]:
                print(f"\n{'='*60}")
                print(f"GPU Transform Level: {level_name}")
                print(f"{'='*60}")

                for batch_size, num_workers, pin_memory in gpu_configs:
                    try:
                        # Clear caches between runs
                        self.clear_caches()

                        result = self.benchmark_gpu_pipeline(
                            transform_level=level_name,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                        )

                        if result:
                            results["gpu_results"].append(asdict(result))
                            print(
                                f"  → GPU Util: {result.gpu_util_percent:.1f}%, "
                                f"Pipeline Eff: {result.pipeline_efficiency:.2f}, "
                                f"Transfer: {result.transfer_bandwidth_gbps:.1f} GB/s"
                            )

                            # Diagnose if utilization is low
                            if result.gpu_util_percent < 10:
                                self.diagnose_low_gpu_utilization(result)

                    except Exception as e:
                        print(f"  ✗ Error: {e}")
                        import traceback

                        traceback.print_exc()

            # Run profiler on best config
            print("\nPhase 4: Detailed Profiling")
            try:
                self.benchmark_with_profiler("level8_production_training", 64, 4)
            except Exception as e:
                print(f"Profiling failed: {e}")

        else:
            print("\n  No GPU available - skipping GPU benchmarks")

        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        """Print comprehensive summary"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print("=" * 80)

        # Component summary
        print("\nComponent Performance:")
        print("-" * 50)
        for comp in results["component_results"]:
            print(
                f"{comp['operation']:<25} {comp['throughput']:>10.1f} {comp['unit']:<10} "
                f"({comp['avg_time_ms']:>6.2f}ms)"
            )

        # CPU Pipeline summary
        print("\nCPU Pipeline Performance (images/sec):")
        print("-" * 80)

        # Group by level
        by_level = defaultdict(list)
        for r in results["pipeline_results"]:
            by_level[r["transform_level"]].append(r)

        # Figure out all the (batch_size, num_workers) combos we ran
        combos = sorted(
            {(r["batch_size"], r["num_workers"]) for r in results["pipeline_results"]},
            key=lambda x: (x[0], x[1]),
        )

        # Print header row
        header = ["Transform".ljust(30)]
        for bs, w in combos:
            if w == 0:
                header.append(f"BS={bs}".center(12))
            else:
                header.append(f"BS={bs},W={w}".center(12))
        print(" | ".join(header))
        print("-" * (32 + len(combos) * 15))

        # Print each level
        for level in sorted(by_level.keys()):
            row = [level.ljust(30)]
            for bs, w in combos:
                hit = next(
                    (
                        r
                        for r in by_level[level]
                        if r["batch_size"] == bs and r["num_workers"] == w
                    ),
                    None,
                )
                if hit:
                    row.append(f"{hit['images_per_sec']:>12.1f}")
                else:
                    row.append("     N/A    ")
            print(" | ".join(row))

        # Thread scaling efficiency summary
        print("\nThread Scaling Efficiency:")
        print("-" * 50)
        for level, results_list in by_level.items():
            for result in results_list:
                if result.get("thread_scaling_efficiency") is not None:
                    efficiency = result["thread_scaling_efficiency"] * 100
                    print(f"{level:<30} {efficiency:>6.1f}% ({result['num_workers']}W)")
                    break  # Only show one efficiency per level

        # GPU Pipeline Summary
        if results.get("gpu_results"):
            print("\nGPU Pipeline Performance:")
            print("-" * 110)
            print(
                f"{'Transform':<25} {'Config':<20} {'GPU%':<8} {'Eff':<8} "
                f"{'Load ms':<10} {'Xfer ms':<10} {'GPU ms':<10} {'BW GB/s':<10}"
            )
            print("-" * 110)

            for r in results["gpu_results"]:
                config = f"B={r['batch_size']},W={r['num_workers']},P={'Y' if r['pin_memory'] else 'N'}"
                print(
                    f"{r['transform_level']:<25} {config:<20} "
                    f"{r['gpu_util_percent']:>6.1f}% "
                    f"{r['pipeline_efficiency']:>7.2f} "
                    f"{r['dataload_time_ms']:>9.1f} "
                    f"{r['transfer_time_ms']:>9.1f} "
                    f"{r['compute_time_ms']:>8.1f} "
                    f"{r['transfer_bandwidth_gbps']:>9.1f}"
                )

    def validate_gpu_setup(self):
        """Comprehensive GPU validation and diagnostics"""
        print("\n" + "=" * 60)
        print("GPU VALIDATION AND DIAGNOSTICS")
        print("=" * 60)

        # 1. Basic CUDA availability
        print(f"1. CUDA Available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("   ✗ CUDA not available - GPU benchmarks will be skipped")
            return False

        # 2. GPU information
        print(f"2. GPU Device: {torch.cuda.get_device_name()}")
        print(f"   - CUDA Version: {torch.version.cuda}")
        print(f"   - Device Count: {torch.cuda.device_count()}")

        # 3. Memory information
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   - Total Memory: {total_memory:.1f} GB")

        # 4. Test basic GPU operations
        print("3. Testing basic GPU operations...")
        try:
            # Test tensor creation and movement
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            print("   ✓ Tensor operations working")

            # Test model inference
            if self.model is not None:
                dummy_input = torch.randn(4, 3, 224, 224, device="cuda")
                with torch.no_grad():
                    output = self.model(dummy_input)
                torch.cuda.synchronize()
                print(f"   ✓ Model inference working (output shape: {output.shape})")

            del x, y, z, dummy_input, output
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ✗ GPU operations failed: {e}")
            return False

        # 5. Test NVML monitoring
        print("4. Testing GPU monitoring...")
        if GPU_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                # Test utilization query
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                print(f"   ✓ NVML working - current GPU util: {util.gpu}%")

                # Test memory query
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"   ✓ Memory monitoring - used: {mem.used/1e9:.1f} GB")

                # Test other queries
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    print(f"   ✓ Temperature monitoring - current: {temp}°C")
                except:
                    print("   ⚠ Temperature monitoring not supported")

                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                    print(f"   ✓ Power monitoring - current: {power:.1f}W")
                except:
                    print("   ⚠ Power monitoring not supported")

            except Exception as e:
                print(f"   ✗ NVML monitoring failed: {e}")
                print("   ⚠ GPU utilization will show 0% in benchmarks")
                print("   💡 Try: pip install nvidia-ml-py")
        else:
            print("   ✗ pynvml not available")
            print("   💡 Install with: pip install nvidia-ml-py")

        # 6. Test pinned memory
        print("5. Testing pinned memory...")
        try:
            cpu_tensor = torch.randn(1000, 1000).pin_memory()
            gpu_tensor = cpu_tensor.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            print("   ✓ Pinned memory working")
            del cpu_tensor, gpu_tensor
        except Exception as e:
            print(f"   ✗ Pinned memory failed: {e}")

        print("\nGPU validation complete!")
        return True

    def diagnose_low_gpu_utilization(self, result: GPUBenchmarkResult):
        """Diagnose why GPU utilization might be low"""
        print(f"\n🔍 DIAGNOSING LOW GPU UTILIZATION ({result.gpu_util_percent:.1f}%)")
        print("-" * 60)

        # Check if monitoring is working at all
        if result.gpu_util_percent == 0:
            print("❌ GPU utilization is 0% - monitoring likely failed")
            print("   Possible causes:")
            print("   1. pynvml not installed: pip install nvidia-ml-py")
            print("   2. NVML driver issues")
            print("   3. Permissions issues")
            print("   4. Docker container without proper GPU access")
            return

        # Analyze timing breakdown
        total_time = (
            result.dataload_time_ms + result.transfer_time_ms + result.compute_time_ms
        )
        compute_ratio = result.compute_time_ms / total_time

        print(f"⏱️  Timing Breakdown:")
        print(
            f"   - Data Loading: {result.dataload_time_ms:.1f}ms ({result.dataload_time_ms/total_time*100:.1f}%)"
        )
        print(
            f"   - Transfer:     {result.transfer_time_ms:.1f}ms ({result.transfer_time_ms/total_time*100:.1f}%)"
        )
        print(
            f"   - Compute:      {result.compute_time_ms:.1f}ms ({compute_ratio*100:.1f}%)"
        )

        # Diagnosis based on patterns
        if result.dataload_time_ms > result.compute_time_ms * 2:
            print("🐌 BOTTLENECK: Data loading is slow")
            print(
                "   💡 Try: Increase num_workers, use faster storage, or reduce transform complexity"
            )

        if result.transfer_time_ms > result.compute_time_ms:
            print("🐌 BOTTLENECK: Data transfer is slow")
            print(
                "   💡 Try: Enable pin_memory=True, reduce batch size, or check PCIe bandwidth"
            )

        if compute_ratio < 0.3:
            print("⚠️  Low compute ratio - GPU is underutilized")
            print(
                "   💡 Try: Increase batch size, reduce I/O overhead, or use a more complex model"
            )

        # Memory analysis
        if result.gpu_memory_mb < 1000:  # Less than 1GB used
            print("💾 Low GPU memory usage - could increase batch size")

        # Bandwidth analysis
        if result.transfer_bandwidth_gbps < 10:  # PCIe 3.0 x16 should do ~12GB/s
            print(
                f"🚌 Low transfer bandwidth ({result.transfer_bandwidth_gbps:.1f} GB/s)"
            )
            print(
                "   💡 Try: pin_memory=True, larger batches, or check PCIe configuration"
            )

    def run_gpu_diagnostics(self):
        """Run comprehensive GPU diagnostics before benchmarking"""
        if not self.validate_gpu_setup():
            return False

        # Run a quick diagnostic benchmark
        print("\n6. Running diagnostic benchmark...")
        result = self.benchmark_gpu_pipeline(
            transform_level="level7_production_inference",
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            num_batches=20,
            warmup_batches=5,
        )

        if result:
            self.diagnose_low_gpu_utilization(result)

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive CPU-GPU PyTorch Benchmark"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to ImageNet/Imagenette data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comprehensive_benchmark_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with fewer iterations"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run only CPU benchmarks"
    )
    parser.add_argument(
        "--gpu-only", action="store_true", help="Run only GPU benchmarks (requires GPU)"
    )

    args = parser.parse_args()

    # Verify data path
    if not Path(args.data).exists():
        print(f"Error: Data path {args.data} does not exist")
        return

    # Create and run benchmark
    benchmark = ComprehensiveBenchmark(args.data)
    benchmark.run_gpu_diagnostics()

    if args.quick:
        # Quick test
        print("Running quick test...")

        # Component test
        comp_results = benchmark.benchmark_components()

        # CPU Pipeline test
        result = benchmark.benchmark_transform_pipeline(
            transform_level="level3_resize",
            batch_size=32,
            num_workers=4,
            num_batches=50,  # Reduced for quick test
            warmup_batches=10,
        )

        if result:
            print(f"\nQuick test results:")
            print(f"  I/O: {comp_results[0].throughput:.1f} MB/s")
            print(f"  Decode: {comp_results[1].throughput:.1f} img/s")
            print(f"  CPU Pipeline: {result.images_per_sec:.1f} img/s")

        # Quick GPU test
        if torch.cuda.is_available() and not args.cpu_only:
            gpu_result = benchmark.benchmark_gpu_pipeline(
                transform_level="level7_production_inference",
                batch_size=64,
                num_workers=4,
                pin_memory=True,
                num_batches=20,
            )

            if gpu_result:
                print(f"\nGPU Quick test:")
                print(f"  GPU Utilization: {gpu_result.gpu_util_percent:.1f}%")
                print(f"  Pipeline Efficiency: {gpu_result.pipeline_efficiency:.2f}")
                print(
                    f"  Transfer Bandwidth: {gpu_result.transfer_bandwidth_gbps:.1f} GB/s"
                )

    else:
        # Full benchmark
        benchmark.run_full_benchmark(args.output)


if __name__ == "__main__":
    main()
