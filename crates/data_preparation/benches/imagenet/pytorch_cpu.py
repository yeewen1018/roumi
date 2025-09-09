# benches/imagenet/pytorch_cpu.py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
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


class RigorousBenchmark:
    """Rigorous CPU-only benchmark with component isolation"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.platform = platform.system()

        # Collect sample paths for component benchmarks
        dataset = ImageFolder(self.data_root)
        self.sample_paths = [s[0] for s in dataset.samples[:1000]]  # Use first 1000

        # Define transform levels
        self.transform_levels = self._get_transform_levels()

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

    def run_full_benchmark(self, output_file: str = "pytorch_benchmark_results.json"):
        """Run complete CPU benchmark suite"""

        # Get system info
        total_memory_gb = psutil.virtual_memory().total / (1024**3)

        results = {
            "metadata": {
                "phase": "rigorous_cpu_benchmark",
                "data_root": str(self.data_root),
                "torch_version": torch.__version__,
                "platform": self.platform,
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": total_memory_gb,
            },
            "component_results": [],
            "pipeline_results": [],
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

        # Run pipeline benchmarks
        print("\nPhase 2: Transform Pipelines (Persistent Workers)")

        # Test configurations - UPDATED to match Rust
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
                    import traceback

                    traceback.print_exc()

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
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Component summary
        print("\nComponent Performance:")
        print("-" * 50)
        for comp in results["component_results"]:
            print(
                f"{comp['operation']:<25} {comp['throughput']:>10.1f} {comp['unit']:<10} "
                f"({comp['avg_time_ms']:>6.2f}ms)"
            )

        # Pipeline summary
        print("\nPipeline Performance (images/sec):")
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


def main():
    parser = argparse.ArgumentParser(description="Rigorous PyTorch CPU Benchmark")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to ImageNet/Imagenette data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pytorch_benchmark_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with fewer iterations"
    )

    args = parser.parse_args()

    # Verify data path
    if not Path(args.data).exists():
        print(f"Error: Data path {args.data} does not exist")
        return

    # Create and run benchmark
    benchmark = RigorousBenchmark(args.data)

    if args.quick:
        # Quick test
        print("Running quick test...")

        # Component test
        comp_results = benchmark.benchmark_components()

        # Pipeline test
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
            print(f"  Pipeline: {result.images_per_sec:.1f} img/s")

    else:
        # Full benchmark
        benchmark.run_full_benchmark(args.output)


if __name__ == "__main__":
    main()
