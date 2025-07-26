//! benches/imagenet/rust_cpu.rs
//! To run this script: cargo run --release --bin benchmark_imagenet_rust_cpu -- benches/data/path benches/{benchmark_result_name}.json

use anyhow::Result;
use data_preparation::*;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use sysinfo::{ProcessExt, System, SystemExt};

// ============================================================================
// System Monitoring
// ============================================================================

struct SystemMonitor {
    system: Arc<Mutex<System>>,
    process_pid: sysinfo::Pid,
}

impl SystemMonitor {
    fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let process_pid = sysinfo::get_current_pid().expect("Failed to get current process PID");

        Self {
            system: Arc::new(Mutex::new(system)),
            process_pid,
        }
    }

    fn get_memory_usage_mb(&self) -> f64 {
        let mut system = self.system.lock().unwrap();
        system.refresh_process(self.process_pid);

        if let Some(process) = system.process(self.process_pid) {
            process.memory() as f64 / 1024.0 / 1024.0 // Convert to MB
        } else {
            0.0
        }
    }

    fn get_cpu_usage_percent(&self) -> f64 {
        let mut system = self.system.lock().unwrap();
        system.refresh_cpu();
        system.refresh_process(self.process_pid);

        if let Some(process) = system.process(self.process_pid) {
            process.cpu_usage() as f64
        } else {
            0.0
        }
    }
}

// Global system monitor
static SYSTEM_MONITOR: OnceLock<SystemMonitor> = OnceLock::new();

fn get_memory_usage() -> f64 {
    SYSTEM_MONITOR
        .get_or_init(|| SystemMonitor::new())
        .get_memory_usage_mb()
}

fn get_cpu_usage() -> f64 {
    SYSTEM_MONITOR
        .get_or_init(|| SystemMonitor::new())
        .get_cpu_usage_percent()
}

// ============================================================================
// Benchmark Result Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComponentBenchmarkResult {
    component: String,
    operation: String,
    throughput: f64,
    unit: String,
    avg_time_ms: f64,
    std_time_ms: f64,
    memory_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransformBenchmarkResult {
    transform_level: String,
    batch_size: usize,
    num_workers: usize,

    // Performance metrics
    images_per_sec: f64,
    batches_per_sec: f64,
    avg_batch_ms: f64,
    p50_batch_ms: f64,
    p95_batch_ms: f64,
    p99_batch_ms: f64,
    std_batch_ms: f64,

    // Resource metrics
    cpu_percent: f64,
    memory_mb: f64,
    memory_delta_mb: f64,

    // Scaling efficiency
    thread_scaling_efficiency: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkReport {
    metadata: Metadata,
    component_results: Vec<ComponentBenchmarkResult>,
    pipeline_results: Vec<TransformBenchmarkResult>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Metadata {
    phase: String,
    data_root: String,
    rust_version: String,
    platform: String,
    cpu_count: usize,
    total_memory_gb: f64,
}

// ============================================================================
// Utility Functions
// ============================================================================

fn percentile(data: &mut [f64], p: f64) -> f64 {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (data.len() - 1) as f64).round() as usize;
    data[idx]
}

fn scan_imagefolder(root: &Path) -> Result<Vec<(PathBuf, usize)>> {
    let mut samples = Vec::new();
    let mut class_to_idx = HashMap::new();
    let mut current_class_idx = 0;

    println!("Scanning directory: {}", root.display());

    // Scan for class directories
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            let class_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| anyhow::anyhow!("Invalid class directory name"))?;

            // Skip hidden directories
            if class_name.starts_with('.') {
                continue;
            }

            if !class_to_idx.contains_key(class_name) {
                class_to_idx.insert(class_name.to_string(), current_class_idx);
                current_class_idx += 1;
            }

            let class_idx = class_to_idx[class_name];

            // Scan for images in class directory
            for img_entry in fs::read_dir(&path)? {
                let img_entry = img_entry?;
                let img_path = img_entry.path();

                if img_path.is_file() {
                    if let Some(ext) = img_path.extension() {
                        let ext_lower = ext.to_string_lossy().to_lowercase();
                        if matches!(
                            ext_lower.as_str(),
                            "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "tif"
                        ) {
                            samples.push((img_path, class_idx));
                        }
                    }
                }
            }
        }
    }

    if samples.is_empty() {
        return Err(anyhow::anyhow!(
            "No images found in {}. Expected ImageFolder structure: root/class/image.jpg",
            root.display()
        ));
    }

    println!(
        "Found {} images in {} classes",
        samples.len(),
        class_to_idx.len()
    );
    Ok(samples)
}

// ============================================================================
// Component Benchmarks
// ============================================================================
struct ComponentBenchmarks {
    sample_paths: Vec<PathBuf>,
}

impl ComponentBenchmarks {
    fn new(data_root: &Path) -> Result<Self> {
        let samples = scan_imagefolder(data_root)?;
        let sample_paths: Vec<PathBuf> = samples
            .into_iter()
            .take(1000)
            .map(|(path, _)| path)
            .collect();

        if sample_paths.is_empty() {
            return Err(anyhow::anyhow!("No valid image paths found"));
        }

        println!(
            "Using {} images for component benchmarks",
            sample_paths.len()
        );
        Ok(Self { sample_paths })
    }

    fn benchmark_io_only(&self) -> Result<ComponentBenchmarkResult> {
        let num_samples = self.sample_paths.len().min(500);
        let paths = &self.sample_paths[..num_samples];
        let mut times = Vec::new();

        // Pre-allocate buffer based on max file size
        let max_size = paths
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .max()
            .unwrap_or(0) as usize;
        let mut buffer = vec![0u8; max_size];

        // Warmup
        for path in &paths[..10] {
            let mut file = File::open(path)?;
            let _ = file.read(&mut buffer)?;
        }

        // Benchmark with pre-allocated buffer
        let mut total_bytes = 0u64;
        for path in paths {
            let start = Instant::now();
            let mut file = File::open(path)?;
            let bytes_read = file.read(&mut buffer)?;
            times.push(start.elapsed().as_secs_f64());
            total_bytes += bytes_read as u64;
        }

        let times_ms: Vec<f64> = times.iter().map(|t| t * 1000.0).collect();
        let avg_time_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let std_time_ms = statistical::standard_deviation(&times_ms, Some(avg_time_ms));

        let throughput_mbps = (total_bytes as f64 / (1024.0 * 1024.0)) / times.iter().sum::<f64>();

        Ok(ComponentBenchmarkResult {
            component: "io".to_string(),
            operation: "disk_read".to_string(),
            throughput: throughput_mbps,
            unit: "MB/s".to_string(),
            avg_time_ms,
            std_time_ms,
            memory_mb: get_memory_usage(),
        })
    }

    fn benchmark_decode_only(&self) -> Result<ComponentBenchmarkResult> {
        use data_preparation::transforms::vision::LoadImage;

        let num_samples = self.sample_paths.len().min(500);
        let paths = &self.sample_paths[..num_samples];
        let mut times = Vec::new();

        let loader = LoadImage::new();

        // Warmup
        for path in &paths[..10.min(paths.len())] {
            let _ = loader.apply(path.clone())?;
        }

        // Benchmark
        for path in paths {
            let start = Instant::now();
            let img = loader.apply(path.clone())?;
            let _ = img.to_rgb8(); // Ensure conversion like PyTorch
            times.push(start.elapsed().as_secs_f64());
        }

        let times_ms: Vec<f64> = times.iter().map(|t| t * 1000.0).collect();
        let avg_time_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let std_time_ms = statistical::standard_deviation(&times_ms, Some(avg_time_ms));
        let throughput = times.len() as f64 / times.iter().sum::<f64>();

        Ok(ComponentBenchmarkResult {
            component: "decode".to_string(),
            operation: "image_decode".to_string(),
            throughput,
            unit: "images/s".to_string(),
            avg_time_ms,
            std_time_ms,
            memory_mb: get_memory_usage(),
        })
    }

    fn benchmark_transform_operations(&self) -> Result<Vec<ComponentBenchmarkResult>> {
        use data_preparation::transforms::vision::*;

        let mut results = Vec::new();

        // Load test images and ensure they're all RGB
        let test_images: Vec<DynamicImage> = self.sample_paths[..100.min(self.sample_paths.len())]
            .iter()
            .map(|p| {
                let img = image::open(p).unwrap();
                match img {
                    DynamicImage::ImageRgb8(_) => img,
                    _ => DynamicImage::ImageRgb8(img.to_rgb8()),
                }
            })
            .collect();

        // Resize benchmark
        let resize_128 = Resize::new((128, 128), image::imageops::FilterType::Triangle)?;
        let times = self.benchmark_single_transform(&test_images, resize_128)?;
        results.push(self.create_result("resize_128x128", times));

        let resize = Resize::new((256, 256), image::imageops::FilterType::Triangle)?;
        let times = self.benchmark_single_transform(&test_images, resize)?;
        results.push(self.create_result("resize_256x256", times));

        let resize_512 = Resize::new((512, 512), image::imageops::FilterType::Triangle)?;
        let times = self.benchmark_single_transform(&test_images, resize_512)?;
        results.push(self.create_result("resize_512x512", times));

        // CenterCrop benchmark
        let crop = CenterCrop::new(224, 224, None)?;
        let resized_images: Vec<_> = test_images
            .iter()
            .map(|img| {
                let resized = Resize::new((256, 256), image::imageops::FilterType::Triangle)
                    .unwrap()
                    .apply(img.clone())
                    .unwrap();
                match resized {
                    DynamicImage::ImageRgb8(_) => resized,
                    _ => DynamicImage::ImageRgb8(resized.to_rgb8()),
                }
            })
            .collect();
        let times = self.benchmark_single_transform(&resized_images, crop)?;
        results.push(self.create_result("center_crop_224", times));

        // RandomCrop benchmark
        dataloader::init_worker_rng(0, 0, 42);
        let random_crop = RandomCrop::new((224, 224))?.with_pad_if_needed(true);
        let times = self.benchmark_single_transform(&test_images, random_crop)?;
        results.push(self.create_result("random_crop_224", times));

        // RandomResizedCrop benchmark
        let random_resized_crop = RandomResizedCrop::new(224, 224)?
            .with_scale((0.08, 1.0))?
            .with_ratio((0.75, 1.333))?;
        let times = self.benchmark_single_transform(&test_images, random_resized_crop)?;
        results.push(self.create_result("random_resized_crop_224", times));

        // RandomHorizontalFlip benchmark
        let horizontal_flip = RandomHorizontalFlip::new(0.5)?;
        let times = self.benchmark_single_transform(&test_images, horizontal_flip)?;
        results.push(self.create_result("random_horizontal_flip", times));

        // RandomRotation benchmark
        dataloader::init_worker_rng(0, 0, 42);
        let random_rotation = RandomRotation::new(15.0)?;
        let times = self.benchmark_single_transform(&test_images, random_rotation)?;
        results.push(self.create_result("random_rotation_15deg", times));

        // Light ColorJitter (common in inference augmentation)
        let color_jitter_light = ColorJitter::new(0.1, 0.1, 0.1, 0.05)?;
        let times = self.benchmark_single_transform(&test_images, color_jitter_light)?;
        results.push(self.create_result("color_jitter_light", times));

        // ColorJitter benchmark
        let color_jitter = ColorJitter::new(0.4, 0.4, 0.4, 0.1)?;
        let times = self.benchmark_single_transform(&test_images, color_jitter)?;
        results.push(self.create_result("color_jitter", times));

        // ToTensor benchmark
        let to_tensor = ToTensor;
        let times = self.benchmark_single_transform(&resized_images, to_tensor)?;
        results.push(self.create_result("to_tensor", times));

        // Normalize benchmark
        let normalize = Normalize::imagenet();
        let times = {
            let mut times = Vec::new();
            for img in &resized_images {
                let tensor = ToTensor.apply(img.clone())?;
                let start = Instant::now();
                let _ = normalize.apply(tensor)?;
                times.push(start.elapsed().as_secs_f64());
            }
            times
        };
        results.push(self.create_result("normalize", times));

        Ok(results)
    }

    fn benchmark_single_transform<I, O, T>(&self, inputs: &[I], transform: T) -> Result<Vec<f64>>
    where
        T: Transform<I, O>,
        I: Clone,
    {
        let mut times = Vec::new();

        for input in inputs {
            let start = Instant::now();
            let _ = transform.apply(input.clone())?;
            times.push(start.elapsed().as_secs_f64());
        }

        Ok(times)
    }

    fn create_result(&self, operation: &str, times: Vec<f64>) -> ComponentBenchmarkResult {
        let times_ms: Vec<f64> = times.iter().map(|t| t * 1000.0).collect();
        let avg_time_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let std_time_ms = statistical::standard_deviation(&times_ms, Some(avg_time_ms));
        let throughput = times.len() as f64 / times.iter().sum::<f64>();

        ComponentBenchmarkResult {
            component: "transform".to_string(),
            operation: operation.to_string(),
            throughput,
            unit: "ops/s".to_string(),
            avg_time_ms,
            std_time_ms,
            memory_mb: get_memory_usage(),
        }
    }
}

// ============================================================================
// Transform Pipeline Benchmarks
// ============================================================================

struct TransformPipelineBenchmark {
    #[allow(dead_code)]
    data_root: PathBuf,
    samples: Vec<(PathBuf, usize)>,
}

impl TransformPipelineBenchmark {
    fn new(data_root: PathBuf) -> Result<Self> {
        let samples = scan_imagefolder(&data_root)?;
        Ok(Self { data_root, samples })
    }

    fn get_transform_levels(
        &self,
    ) -> HashMap<String, Box<dyn Fn() -> Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>>>
    {
        use data_preparation::transforms::vision::*;
        use image::imageops::FilterType;

        let mut levels = HashMap::new();

        // Level 2: Tensor only
        levels.insert(
            "level2_tensor_only".to_string(),
            Box::new(|| {
                let pipeline = EnsureRGB
                    .then(Resize::new((224, 224), FilterType::Triangle).unwrap())
                    .then(ToTensor);
                Box::new(LoadImageToSample::new(pipeline))
                    as Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>
            })
                as Box<dyn Fn() -> Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>>,
        );

        // Level 3: Basic resize
        levels.insert(
            "level3_resize".to_string(),
            Box::new(|| {
                let pipeline = EnsureRGB
                    .then(Resize::new((256, 256), FilterType::Triangle).unwrap())
                    .then(CenterCrop::new(224, 224, None).unwrap())
                    .then(ToTensor);
                Box::new(LoadImageToSample::new(pipeline))
                    as Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>
            })
                as Box<dyn Fn() -> Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>>,
        );

        // Level 4: Basic augmentation
        levels.insert(
            "level4_augment_basic".to_string(),
            Box::new(|| {
                let pipeline = EnsureRGB
                    .then(RandomResizedCrop::new(224, 224).unwrap())
                    .then(RandomHorizontalFlip::new(0.5).unwrap())
                    .then(ToTensor);
                Box::new(LoadImageToSample::new(pipeline))
                    as Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>
            })
                as Box<dyn Fn() -> Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>>,
        );

        // Level 5: Full augmentation
        levels.insert(
            "level5_augment_full".to_string(),
            Box::new(|| {
                let pipeline = EnsureRGB
                    .then(RandomResizedCrop::new(224, 224).unwrap())
                    .then(RandomHorizontalFlip::new(0.5).unwrap())
                    .then(ColorJitter::new(0.4, 0.4, 0.4, 0.1).unwrap())
                    .then(ToTensor);
                Box::new(LoadImageToSample::new(pipeline))
                    as Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>
            })
                as Box<dyn Fn() -> Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>>,
        );

        // Level 6: With normalization
        levels.insert(
            "level6_normalize".to_string(),
            Box::new(|| {
                let pipeline = EnsureRGB
                    .then(Resize::new((256, 256), FilterType::Triangle).unwrap())
                    .then(CenterCrop::new(224, 224, None).unwrap())
                    .then(ToTensor)
                    .then(Normalize::imagenet());
                Box::new(LoadImageToSample::new(pipeline))
                    as Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>
            })
                as Box<dyn Fn() -> Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>>,
        );

        // Level 7: Production inference
        levels.insert(
            "level7_production_inference".to_string(),
            Box::new(|| {
                let pipeline = EnsureRGB
                    .then(Resize::new((256, 256), FilterType::Triangle).unwrap())
                    .then(CenterCrop::new(224, 224, None).unwrap())
                    .then(ToTensor)
                    .then(Normalize::imagenet());
                Box::new(LoadImageToSample::new(pipeline))
                    as Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>
            })
                as Box<dyn Fn() -> Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>>,
        );

        // Level 8: Production training
        levels.insert(
            "level8_production_training".to_string(),
            Box::new(|| {
                let pipeline = EnsureRGB
                    .then(RandomResizedCrop::new(224, 224).unwrap())
                    .then(RandomHorizontalFlip::new(0.5).unwrap())
                    .then(ColorJitter::new(0.4, 0.4, 0.4, 0.1).unwrap())
                    .then(RandomRotation::new(15.0).unwrap())
                    .then(ToTensor)
                    .then(Normalize::imagenet());
                Box::new(LoadImageToSample::new(pipeline))
                    as Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>
            })
                as Box<dyn Fn() -> Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>>,
        );

        levels
    }

    fn benchmark_pipeline(
        &self,
        transform_level: &str,
        transform_creator: &dyn Fn() -> Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>,
        batch_size: usize,
        num_workers: usize,
        num_batches: usize,
        warmup_batches: usize,
        single_threaded_baseline: Option<f64>,
    ) -> Result<TransformBenchmarkResult> {
        println!(
            "\nBenchmarking: {}, batch_size={}, workers={}",
            transform_level, batch_size, num_workers
        );

        // Create transform
        let transform = transform_creator();

        // Create dataset
        let dataset = InMemoryDataset::new(self.samples.clone()).with_transform(transform);

        println!("Dataset: {} images", dataset.len());

        // Create config
        let config = DataLoaderConfig::builder()
            .batch_size(batch_size)
            .shuffle(true)
            .num_workers(num_workers)
            .persistent_workers(true) // Fixed to persistent workers
            .prefetch_factor(2) // Fixed prefetch factor
            .drop_last(true)
            .seed(42)
            .build();

        // Create loader
        let loader = DataLoader::new(dataset, config)?;

        // Memory baseline
        let mem_before = get_memory_usage();

        // Create iterator once
        let mut iter = loader.iter()?;

        // Measure startup time (first batch includes worker startup)
        let startup_start = Instant::now();
        let _ = iter.next().ok_or_else(|| anyhow::anyhow!("No batches"))?;
        let startup_time = startup_start.elapsed();
        println!("  Worker startup time: {:.2}s", startup_time.as_secs_f64());

        // Warmup
        println!("  Warming up with {} batches...", warmup_batches);
        for _i in 1..warmup_batches {
            if iter.next().is_none() {
                break;
            }
        }

        // Benchmark
        println!("  Benchmarking {} batches...", num_batches);
        let mut batch_times = Vec::new();
        let mut cpu_samples = Vec::new();

        let start_time = Instant::now();

        for i in 0..num_batches {
            let batch_start = Instant::now();

            match iter.next() {
                Some(Ok(batch)) => {
                    // Force evaluation and validation
                    assert_eq!(batch.batch_size()?, batch_size as i64);
                }
                Some(Err(e)) => return Err(e),
                None => {
                    println!("  Warning: Only got {} batches", i);
                    break;
                }
            }

            let batch_time = batch_start.elapsed();
            batch_times.push(batch_time.as_secs_f64());

            // Sample CPU every 10 batches
            if i % 10 == 0 {
                cpu_samples.push(get_cpu_usage());
                if i % 50 == 0 || i < 10 {
                    println!(
                        "    Batch {}/{}: {:.1}ms",
                        i,
                        num_batches,
                        batch_time.as_secs_f64() * 1000.0
                    );
                }
            }
        }

        // Drop the iterator to ensure cleanup
        drop(iter);

        let total_time = start_time.elapsed();
        let mem_after = get_memory_usage();

        // Calculate thread scaling efficiency
        let thread_scaling_efficiency = if let Some(baseline) = single_threaded_baseline {
            if num_workers > 0 {
                let actual_speedup = baseline / (total_time.as_secs_f64() / num_batches as f64);
                Some(actual_speedup / num_workers as f64)
            } else {
                None
            }
        } else {
            None
        };

        // Calculate statistics
        let mut batch_times_ms: Vec<f64> = batch_times.iter().map(|t| t * 1000.0).collect();

        let total_images = batch_times.len() * batch_size;
        let images_per_sec = total_images as f64 / total_time.as_secs_f64();
        let batches_per_sec = batch_times.len() as f64 / total_time.as_secs_f64();

        let avg_batch_ms = batch_times_ms.iter().sum::<f64>() / batch_times_ms.len() as f64;
        let std_batch_ms = statistical::standard_deviation(&batch_times_ms, Some(avg_batch_ms));

        let p50 = percentile(&mut batch_times_ms, 50.0);
        let p95 = percentile(&mut batch_times_ms, 95.0);
        let p99 = percentile(&mut batch_times_ms, 99.0);

        let cpu_percent = if cpu_samples.is_empty() {
            0.0
        } else {
            cpu_samples.iter().sum::<f64>() / cpu_samples.len() as f64
        };

        println!("  → {:.1} img/s, {:.1}ms avg", images_per_sec, avg_batch_ms);
        if let Some(efficiency) = thread_scaling_efficiency {
            println!("  → Thread scaling efficiency: {:.1}%", efficiency * 100.0);
        }

        Ok(TransformBenchmarkResult {
            transform_level: transform_level.to_string(),
            batch_size,
            num_workers,
            images_per_sec,
            batches_per_sec,
            avg_batch_ms,
            p50_batch_ms: p50,
            p95_batch_ms: p95,
            p99_batch_ms: p99,
            std_batch_ms,
            cpu_percent,
            memory_mb: mem_after,
            memory_delta_mb: mem_after - mem_before,
            thread_scaling_efficiency,
        })
    }
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

fn run_benchmark(data_root: PathBuf, output_file: &str) -> Result<()> {
    let mut system = System::new_all();
    system.refresh_all();
    let total_memory_gb = system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

    let mut report = BenchmarkReport {
        metadata: Metadata {
            phase: "rigorous_cpu_benchmark".to_string(),
            data_root: data_root.to_string_lossy().to_string(),
            rust_version: env!("CARGO_PKG_VERSION").to_string(),
            platform: std::env::consts::OS.to_string(),
            cpu_count: num_cpus::get(),
            total_memory_gb,
        },
        component_results: Vec::new(),
        pipeline_results: Vec::new(),
    };

    // Platform warning
    if cfg!(target_os = "macos") {
        println!("\n⚠️  Note: On macOS, multiprocessing may have overhead differences vs Linux.");
        println!("   Results may vary compared to PyTorch due to platform differences.\n");
    }

    // Phase 1: Component benchmarks
    println!("\n{}", "=".repeat(80));
    println!("PHASE 1: COMPONENT ISOLATION BENCHMARKS");
    println!("{}", "=".repeat(80));

    let component_bench = ComponentBenchmarks::new(&data_root)?;

    // I/O benchmark
    println!("\n1. Pure I/O (disk read only)");
    let io_result = component_bench.benchmark_io_only()?;
    println!(
        "   → {:.1} {}, {:.2}ms avg",
        io_result.throughput, io_result.unit, io_result.avg_time_ms
    );
    report.component_results.push(io_result);

    // Decode benchmark
    println!("\n2. I/O + Image Decode");
    let decode_result = component_bench.benchmark_decode_only()?;
    println!(
        "   → {:.1} {}, {:.2}ms avg",
        decode_result.throughput, decode_result.unit, decode_result.avg_time_ms
    );
    report.component_results.push(decode_result);

    // Transform operations
    println!("\n3. Transform Operations");
    let transform_results = component_bench.benchmark_transform_operations()?;
    for result in transform_results {
        println!(
            "   {}: {:.1} {}, {:.2}ms avg",
            result.operation, result.throughput, result.unit, result.avg_time_ms
        );
        report.component_results.push(result);
    }

    // Phase 2: Pipeline benchmarks
    println!("\n{}", "=".repeat(80));
    println!("PHASE 2: TRANSFORM PIPELINES (PERSISTENT WORKERS)");
    println!("{}", "=".repeat(80));

    let pipeline_bench = TransformPipelineBenchmark::new(data_root)?;
    let transforms = pipeline_bench.get_transform_levels();

    // Test configurations: batch_size, num_workers
    let configs = vec![
        (32, 0), // Single-threaded baseline
        (32, 4), // Multi-threaded
        (64, 4), // Larger batch
    ];

    let num_batches = 300;
    let warmup_batches = 30;

    for (level_name, transform_creator) in &transforms {
        println!("\n{}", "=".repeat(60));
        println!("Transform Level: {}", level_name);
        println!("{}", "=".repeat(60));

        let mut single_threaded_time: Option<f64> = None;

        for (batch_size, num_workers) in &configs {
            match pipeline_bench.benchmark_pipeline(
                level_name,
                transform_creator.as_ref(),
                *batch_size,
                *num_workers,
                num_batches,
                warmup_batches,
                single_threaded_time,
            ) {
                Ok(result) => {
                    // Store single-threaded baseline for scaling efficiency calculation
                    if *num_workers == 0 {
                        single_threaded_time = Some(result.avg_batch_ms / 1000.0);
                    }

                    report.pipeline_results.push(result);
                }
                Err(e) => {
                    println!("  ✗ Error: {}", e);
                }
            }
        }
    }

    // Save results
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(output_file, json)?;
    println!("\nResults saved to {}", output_file);

    // Print summary
    print_summary(&report);

    Ok(())
}

fn print_summary(report: &BenchmarkReport) {
    println!("\n{}", "=".repeat(80));
    println!("BENCHMARK SUMMARY");
    println!("{}", "=".repeat(80));

    // Component summary
    println!("\nComponent Performance:");
    println!("{}", "-".repeat(50));
    for comp in &report.component_results {
        println!(
            "{:<25} {:>10.1} {:<10} ({:>6.2}ms)",
            comp.operation, comp.throughput, comp.unit, comp.avg_time_ms
        );
    }

    // Pipeline summary
    println!("\nPipeline Performance (images/sec):");
    println!("{}", "-".repeat(80));

    let configs = vec![(32, 0), (32, 4), (64, 4)];

    // Group by level
    use std::collections::BTreeMap;
    let mut by_level: BTreeMap<String, Vec<&TransformBenchmarkResult>> = BTreeMap::new();
    for result in &report.pipeline_results {
        by_level
            .entry(result.transform_level.clone())
            .or_default()
            .push(result);
    }

    // Print dynamic header
    print!("{:<30}", "Transform");
    for &(bs, w) in &configs {
        if w == 0 {
            print!(" | {:>12}", format!("BS={}", bs));
        } else {
            print!(" | {:>12}", format!("BS={},W={}", bs, w));
        }
    }
    println!();
    println!("{}", "-".repeat(30 + configs.len() * 15));

    // Print each row
    for (level, results) in &by_level {
        print!("{:<30}", level);
        for &(bs, w) in &configs {
            if let Some(r) = results
                .iter()
                .find(|r| r.batch_size == bs && r.num_workers == w)
            {
                print!(" | {:>12.1}", r.images_per_sec);
            } else {
                print!(" | {:>12}", "N/A");
            }
        }
        println!();
    }

    // Thread scaling efficiency summary
    println!("\nThread Scaling Efficiency:");
    println!("{}", "-".repeat(50));
    for (level, results) in by_level {
        for result in results {
            if let Some(efficiency) = result.thread_scaling_efficiency {
                println!(
                    "{:<30} {:>6.1}% ({}W)",
                    level,
                    efficiency * 100.0,
                    result.num_workers
                );
                break; // Only show one efficiency per level
            }
        }
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Filter out cargo bench arguments
    let filtered_args: Vec<String> = args
        .into_iter()
        .filter(|arg| !arg.starts_with("--bench") && !arg.starts_with("--test"))
        .collect();

    if filtered_args.len() < 2 {
        eprintln!("Usage: {} <data_path> [output_file]", filtered_args[0]);
        std::process::exit(1);
    }

    let data_path = PathBuf::from(&filtered_args[1]);
    let output_file = filtered_args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("rust_benchmark_results.json");

    // Verify data path
    if !data_path.exists() {
        eprintln!("Error: Data path {:?} does not exist", data_path);
        std::process::exit(1);
    }

    run_benchmark(data_path, output_file)?;

    Ok(())
}

// Helper module for statistics
mod statistical {
    pub fn standard_deviation(data: &[f64], mean: Option<f64>) -> f64 {
        let mean = mean.unwrap_or_else(|| data.iter().sum::<f64>() / data.len() as f64);
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }
}
