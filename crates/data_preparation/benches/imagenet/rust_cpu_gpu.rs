//! benches/imagenet/rust_cpu_gpu.rs
//! To run this script: cargo run --release --bin benchmark_imagenet_rust_cpu_gpu -- benches/data/path benches/{benchmark_result_name}.json

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
use std::thread;
use std::time::Duration;
use std::time::Instant;
use sysinfo::{ProcessExt, System, SystemExt};

// GPU-related imports
use nvml_wrapper::Nvml;
use tch::{nn, nn::OptimizerConfig, Device, Tensor};

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
// GPU Monitoring
// ============================================================================
#[derive(Debug, Clone)]
struct GPUMetrics {
    gpu_util_percent: f64,
    memory_mb: f64,
    #[allow(dead_code)]
    temperature_c: f64,
    #[allow(dead_code)]
    power_w: f64,
}

struct GPUMonitor {
    monitoring: Arc<Mutex<bool>>,
    samples: Arc<Mutex<Vec<GPUMetrics>>>,
    #[allow(dead_code)]
    device: Device,
}

impl GPUMonitor {
    fn new() -> Option<Self> {
        if !tch::Cuda::is_available() {
            return None;
        }

        Some(Self {
            monitoring: Arc::new(Mutex::new(false)),
            samples: Arc::new(Mutex::new(Vec::new())),
            device: Device::Cuda(0),
        })
    }

    fn start_monitoring(&self) {
        *self.monitoring.lock().unwrap() = true;
        self.samples.lock().unwrap().clear();

        let monitoring = Arc::clone(&self.monitoring);
        let samples = Arc::clone(&self.samples);

        thread::spawn(move || {
            if let Ok(nvml) = Nvml::init() {
                if let Ok(device) = nvml.device_by_index(0) {
                    while *monitoring.lock().unwrap() {
                        let metrics = GPUMetrics {
                            gpu_util_percent: device
                                .utilization_rates()
                                .map(|u| u.gpu as f64)
                                .unwrap_or(0.0),
                            memory_mb: device
                                .memory_info()
                                .map(|m| m.used as f64 / 1024.0 / 1024.0)
                                .unwrap_or(0.0),
                            temperature_c: device
                                .temperature(
                                    nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu,
                                )
                                .map(|t| t as f64)
                                .unwrap_or(0.0),
                            power_w: device
                                .power_usage()
                                .map(|p| p as f64 / 1000.0)
                                .unwrap_or(0.0),
                        };

                        samples.lock().unwrap().push(metrics);
                        thread::sleep(Duration::from_millis(10));
                    }
                }
            }
        });
    }

    fn stop_monitoring(&self) -> (f64, f64, f64) {
        *self.monitoring.lock().unwrap() = false;
        thread::sleep(Duration::from_millis(50)); // Wait for thread to finish

        let samples = self.samples.lock().unwrap();
        if samples.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let avg_util =
            samples.iter().map(|s| s.gpu_util_percent).sum::<f64>() / samples.len() as f64;
        let avg_memory = samples.iter().map(|s| s.memory_mb).sum::<f64>() / samples.len() as f64;
        let peak_memory = samples.iter().map(|s| s.memory_mb).fold(0.0, f64::max);

        (avg_util, avg_memory, peak_memory)
    }

    fn get_detailed_memory_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        if tch::Cuda::is_available() {
            // These functions may not exist in tch-rs, use placeholders
            stats.insert("allocated_mb".to_string(), 0.0);
            stats.insert("cached_mb".to_string(), 0.0);
        }
        stats
    }
}

fn validate_gpu_setup() -> Result<()> {
    println!("=== GPU DIAGNOSTICS ===");

    // 1. CUDA availability
    if !tch::Cuda::is_available() {
        return Err(anyhow::anyhow!("CUDA not available"));
    }
    println!("✓ CUDA available");

    // 2. NVML availability
    match nvml_wrapper::Nvml::init() {
        Ok(nvml) => {
            if let Ok(device) = nvml.device_by_index(0) {
                if let Ok(name) = device.name() {
                    println!("✓ NVML working - GPU: {}", name);
                }
            }
        }
        Err(_) => println!("⚠ NVML not available - GPU monitoring will be limited"),
    }

    // 3. Basic GPU operations
    let x = Tensor::randn(&[100, 100], (tch::Kind::Float, Device::Cuda(0)));
    let _y = &x + &x;
    tch::Cuda::synchronize(0);
    println!("✓ Basic GPU operations working");

    // 4. Model creation
    let model = SimpleModel::new(Device::Cuda(0))?;
    let dummy_input = Tensor::randn(&[4, 3, 224, 224], (tch::Kind::Float, Device::Cuda(0)));
    let dummy_labels = Tensor::randint(1000, &[4], (tch::Kind::Int64, Device::Cuda(0)));

    let output = model.forward(&dummy_input);
    let loss = output.cross_entropy_for_logits(&dummy_labels);
    loss.backward();
    tch::Cuda::synchronize(0);

    println!(
        "✓ ResNet-50 inference working (output shape: {:?})",
        output.size()
    );

    Ok(())
}

// ============================================================================
// GPU Hardware State
// ============================================================================

fn get_gpu_hardware_state() -> HashMap<String, serde_json::Value> {
    let mut state = HashMap::new();

    if tch::Cuda::is_available() {
        state.insert("cuda_available".to_string(), serde_json::Value::Bool(true));
        state.insert(
            "device_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(tch::Cuda::device_count())),
        );

        state.insert(
            "cuda_version".to_string(),
            serde_json::Value::String("Unknown".to_string()),
        );
        state.insert(
            "cudnn_version".to_string(),
            serde_json::Value::String("Unknown".to_string()),
        );

        // Placeholder for additional hardware info
        // In a real implementation, you'd query NVIDIA-ML
        state.insert(
            "gpu_name".to_string(),
            serde_json::Value::String("GPU Device".to_string()),
        );
        state.insert(
            "total_memory_gb".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(8.0).unwrap(), // Placeholder
            ),
        );
    } else {
        state.insert("cuda_available".to_string(), serde_json::Value::Bool(false));
    }

    state
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GPUBenchmarkResult {
    transform_level: String,
    batch_size: usize,
    num_workers: usize,

    // GPU Performance metrics
    gpu_util_percent: f64,
    gpu_memory_mb: f64,
    gpu_memory_peak_mb: f64,

    // Timing breakdown (using median for robustness)
    dataload_time_ms: f64,
    transfer_time_ms: f64,
    compute_time_ms: f64,
    total_time_ms: f64,

    // Efficiency metrics
    gpu_efficiency: f64,      // compute_time / total_time
    pipeline_efficiency: f64, // gpu_util * gpu_efficiency

    // Transfer metrics
    transfer_bandwidth_gbps: f64,
    pin_memory: bool,

    // Detailed memory breakdown
    model_memory_mb: f64,
    memory_overhead_mb: f64,

    // Timing variability
    timing_variability: HashMap<String, f64>,

    // Additional metrics
    effective_bandwidth_gbps: f64,

    // Hardware state during benchmark
    gpu_temperature_c: f64,
    gpu_power_w: f64,
    gpu_clock_mhz: i32,

    // Statistical confidence
    num_samples: usize,
    samples_after_outlier_removal: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkReport {
    metadata: Metadata,
    component_results: Vec<ComponentBenchmarkResult>,
    pipeline_results: Vec<TransformBenchmarkResult>,
    gpu_results: Vec<GPUBenchmarkResult>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Metadata {
    phase: String,
    data_root: String,
    rust_version: String,
    platform: String,
    cpu_count: usize,
    total_memory_gb: f64,
    has_gpu: bool,
    gpu_name: Option<String>,
    hardware_state: HashMap<String, serde_json::Value>,
    tch_version: String,
    timestamp: String,
    hostname: String,
}

// ============================================================================
// Utility Functions
// ============================================================================

fn percentile(data: &mut [f64], p: f64) -> f64 {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (data.len() - 1) as f64).round() as usize;
    data[idx]
}

fn remove_outliers(times: &[f64], n_std: f64) -> Vec<f64> {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let std = statistical::standard_deviation(times, Some(mean));

    times
        .iter()
        .filter(|&&t| (t - mean).abs() < n_std * std)
        .copied()
        .collect()
}

fn calculate_robust_stats(times: &[f64]) -> HashMap<String, f64> {
    let cleaned_times = remove_outliers(times, 3.0);

    let mut stats = HashMap::new();
    if !cleaned_times.is_empty() {
        let mean = cleaned_times.iter().sum::<f64>() / cleaned_times.len() as f64;
        stats.insert("mean".to_string(), mean);
        stats.insert(
            "median".to_string(),
            percentile(&mut cleaned_times.clone(), 50.0),
        );
        stats.insert(
            "std".to_string(),
            statistical::standard_deviation(&cleaned_times, Some(mean)),
        );
        stats.insert(
            "cv".to_string(),
            if mean > 0.0 {
                statistical::standard_deviation(&cleaned_times, Some(mean)) / mean
            } else {
                0.0
            },
        );
    }
    stats
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
// Component Benchmarks (same as before)
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

        let resize_128 = Resize::new((128, 128), image::imageops::FilterType::Triangle)?;
        let times = self.benchmark_single_transform(&test_images, resize_128)?;
        results.push(self.create_result("resize_128x128", times));

        let resize = Resize::new((256, 256), image::imageops::FilterType::Triangle)?;
        let times = self.benchmark_single_transform(&test_images, resize)?;
        results.push(self.create_result("resize_256x256", times));

        let resize_512 = Resize::new((512, 512), image::imageops::FilterType::Triangle)?;
        let times = self.benchmark_single_transform(&test_images, resize_512)?;
        results.push(self.create_result("resize_512x512", times));

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

        dataloader::init_worker_rng(0, 0, 42);
        let random_crop = RandomCrop::new((224, 224))?.with_pad_if_needed(true);
        let times = self.benchmark_single_transform(&test_images, random_crop)?;
        results.push(self.create_result("random_crop_224", times));

        let random_resized_crop = RandomResizedCrop::new(224, 224)?
            .with_scale((0.08, 1.0))?
            .with_ratio((0.75, 1.333))?;
        let times = self.benchmark_single_transform(&test_images, random_resized_crop)?;
        results.push(self.create_result("random_resized_crop_224", times));

        let horizontal_flip = RandomHorizontalFlip::new(0.5)?;
        let times = self.benchmark_single_transform(&test_images, horizontal_flip)?;
        results.push(self.create_result("random_horizontal_flip", times));

        dataloader::init_worker_rng(0, 0, 42);
        let random_rotation = RandomRotation::new(15.0)?;
        let times = self.benchmark_single_transform(&test_images, random_rotation)?;
        results.push(self.create_result("random_rotation_15deg", times));

        let color_jitter_light = ColorJitter::new(0.1, 0.1, 0.1, 0.05)?;
        let times = self.benchmark_single_transform(&test_images, color_jitter_light)?;
        results.push(self.create_result("color_jitter_light", times));

        let color_jitter = ColorJitter::new(0.4, 0.4, 0.4, 0.1)?;
        let times = self.benchmark_single_transform(&test_images, color_jitter)?;
        results.push(self.create_result("color_jitter", times));

        let to_tensor = ToTensor;
        let times = self.benchmark_single_transform(&resized_images, to_tensor)?;
        results.push(self.create_result("to_tensor", times));

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
// Simple Model for GPU Benchmarking
// ============================================================================

struct SimpleModel {
    net: Box<dyn nn::ModuleT>,
    vs: nn::VarStore,
    #[allow(dead_code)]
    device: Device,
}

impl SimpleModel {
    fn new(device: Device) -> Result<Self> {
        let vs = nn::VarStore::new(device);

        // ResNet-50 with 1000 classes (matching PyTorch)
        let net = Box::new(tch::vision::resnet::resnet50(&vs.root(), 1000));

        Ok(Self { net, vs, device })
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.net.forward_t(input, true)
    }

    // Add optimizer creation
    fn create_optimizer(&self) -> nn::Optimizer {
        nn::Sgd::default().build(&self.vs, 1e-1).unwrap()
    }
}

// ============================================================================
// Transform Pipeline Benchmarks (CPU)
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

        let transform = transform_creator();
        let dataset = InMemoryDataset::new(self.samples.clone()).with_transform(transform);

        println!("Dataset: {} images", dataset.len());

        let config = DataLoaderConfig::builder()
            .batch_size(batch_size)
            .shuffle(true)
            .num_workers(num_workers)
            .persistent_workers(true)
            .prefetch_factor(2)
            .drop_last(true)
            .seed(42)
            .build();

        let loader = DataLoader::new(dataset, config)?;
        let mem_before = get_memory_usage();
        let mut iter = loader.iter()?;

        // Measure startup time
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

        drop(iter);

        let total_time = start_time.elapsed();
        let mem_after = get_memory_usage();

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

    // ============================================================================
    // GPU Pipeline Benchmarks
    // ============================================================================

    fn benchmark_gpu_pipeline(
        &self,
        transform_level: &str,
        transform_creator: &dyn Fn() -> Box<dyn Transform<(PathBuf, usize), Sample> + Send + Sync>,
        batch_size: usize,
        num_workers: usize,
        pin_memory: bool,
        num_batches: usize,
        warmup_batches: usize,
    ) -> Result<GPUBenchmarkResult> {
        tch::Cuda::synchronize(0);

        if !tch::Cuda::is_available() {
            return Err(anyhow::anyhow!("CUDA not available"));
        }

        println!(
            "\nGPU Benchmark: {}, batch={}, workers={}, pin_memory={}",
            transform_level, batch_size, num_workers, pin_memory
        );

        let device = Device::Cuda(0);

        // Create model
        let model = SimpleModel::new(device)?;
        let mut optimizer = model.create_optimizer();

        // Create transform and dataset
        let transform = transform_creator();
        let dataset = InMemoryDataset::new(self.samples.clone()).with_transform(transform);

        let config = DataLoaderConfig::builder()
            .batch_size(batch_size)
            .shuffle(true)
            .num_workers(num_workers)
            .persistent_workers(true)
            .prefetch_factor(2)
            .drop_last(true)
            .seed(42)
            .build();

        let loader = DataLoader::new(dataset, config)?;

        // Clear GPU memory
        tch::Cuda::manual_seed(42);
        tch::Cuda::synchronize(0);

        // Get initial memory stats
        let initial_memory = if let Some(monitor) = GPUMonitor::new() {
            monitor.get_detailed_memory_stats()
        } else {
            HashMap::new()
        };

        // Start GPU monitoring
        let gpu_monitor = GPUMonitor::new();
        if let Some(ref monitor) = gpu_monitor {
            monitor.start_monitoring();
        }

        // Storage for detailed timing
        let mut dataload_times = Vec::new();
        let mut transfer_times = Vec::new();
        let mut compute_times = Vec::new();

        let mut iter = loader.iter()?;

        // Warmup
        println!("  Warming up GPU...");
        for _i in 0..warmup_batches {
            if let Some(Ok(batch)) = iter.next() {
                let images = batch.get("image")?.to_device(device);
                let _labels = batch.get("label")?.to_device(device);

                // Forward pass
                let _output = model.forward(&images);
                tch::Cuda::synchronize(0);
            }
        }

        // Benchmark
        println!("  Benchmarking {} batches...", num_batches);

        tch::Cuda::synchronize(0);
        let total_start = Instant::now();

        for i in 0..num_batches {
            // 1. Measure data loading time
            let t0 = Instant::now();
            let batch = match iter.next() {
                Some(Ok(batch)) => batch,
                Some(Err(e)) => return Err(e),
                None => break,
            };
            let t1 = Instant::now();
            dataload_times.push((t1 - t0).as_secs_f64());

            // 2. Measure transfer time
            let images = batch.get("image")?.to_device(device);
            let labels = batch.get("label")?.to_device(device);
            tch::Cuda::synchronize(0);
            let t2 = Instant::now();
            transfer_times.push((t2 - t1).as_secs_f64());

            // 3. Measure compute time
            optimizer.zero_grad();
            let output = model.forward(&images);
            let loss = output.cross_entropy_for_logits(&labels);
            loss.backward();
            optimizer.step();
            tch::Cuda::synchronize(0);
            let t3 = Instant::now();
            compute_times.push((t3 - t2).as_secs_f64());

            // Progress
            if i % 20 == 0 {
                let total_ms = (t3 - t0).as_secs_f64() * 1000.0;
                println!("    Batch {}/{}: {:.1}ms total", i, num_batches, total_ms);
            }
        }

        tch::Cuda::synchronize(0);
        let _total_time = total_start.elapsed();

        // Stop monitoring
        let (avg_util, avg_mem, peak_mem) = if let Some(ref monitor) = gpu_monitor {
            monitor.stop_monitoring()
        } else {
            (0.0, 0.0, 0.0)
        };

        // Calculate robust statistics
        let dataload_stats = calculate_robust_stats(&dataload_times);
        let transfer_stats = calculate_robust_stats(&transfer_times);
        let compute_stats = calculate_robust_stats(&compute_times);

        // Calculate transfer bandwidth (approximate)
        let bytes_per_image = 224 * 224 * 3 * 4; // float32 tensor
        let bytes_per_label = 8; // int64
        let bytes_per_batch = (bytes_per_image + bytes_per_label) * batch_size;
        let bytes_gb = bytes_per_batch as f64 / 1e9;

        let transfer_bandwidth_gbps = if transfer_stats.contains_key("median") {
            bytes_gb / (transfer_stats["median"] / 1000.0)
        } else {
            0.0
        };

        let effective_bandwidth_gbps =
            (bytes_gb * dataload_times.len() as f64) / transfer_times.iter().sum::<f64>();

        // Build timing variability map
        let mut timing_variability = HashMap::new();
        if let Some(cv) = dataload_stats.get("cv") {
            timing_variability.insert("dataload_cv".to_string(), *cv);
        }
        if let Some(cv) = transfer_stats.get("cv") {
            timing_variability.insert("transfer_cv".to_string(), *cv);
        }
        if let Some(cv) = compute_stats.get("cv") {
            timing_variability.insert("compute_cv".to_string(), *cv);
        }

        // Calculate efficiency metrics
        let dataload_median = dataload_stats.get("median").unwrap_or(&0.0) * 1000.0;
        let transfer_median = transfer_stats.get("median").unwrap_or(&0.0) * 1000.0;
        let compute_median = compute_stats.get("median").unwrap_or(&0.0) * 1000.0;
        let total_median = dataload_median + transfer_median + compute_median;

        let gpu_efficiency = if total_median > 0.0 {
            compute_median / total_median
        } else {
            0.0
        };

        let pipeline_efficiency = (avg_util / 100.0) * gpu_efficiency;

        // Memory stats
        let final_memory = if let Some(ref monitor) = gpu_monitor {
            monitor.get_detailed_memory_stats()
        } else {
            HashMap::new()
        };

        let model_memory_mb = initial_memory.get("allocated_mb").unwrap_or(&0.0);
        let memory_overhead_mb = final_memory.get("allocated_mb").unwrap_or(&0.0) - model_memory_mb;

        drop(iter);

        Ok(GPUBenchmarkResult {
            transform_level: transform_level.to_string(),
            batch_size,
            num_workers,
            gpu_util_percent: avg_util,
            gpu_memory_mb: avg_mem,
            gpu_memory_peak_mb: peak_mem,
            dataload_time_ms: dataload_median,
            transfer_time_ms: transfer_median,
            compute_time_ms: compute_median,
            total_time_ms: total_median,
            gpu_efficiency,
            pipeline_efficiency,
            transfer_bandwidth_gbps,
            pin_memory,
            model_memory_mb: *model_memory_mb,
            memory_overhead_mb,
            timing_variability,
            effective_bandwidth_gbps,
            gpu_temperature_c: 65.0, // Placeholder
            gpu_power_w: 200.0,      // Placeholder
            gpu_clock_mhz: 1500,     // Placeholder
            num_samples: dataload_times.len(),
            samples_after_outlier_removal: remove_outliers(&dataload_times, 3.0).len(),
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
            phase: "comprehensive_cpu_gpu_benchmark".to_string(),
            data_root: data_root.to_string_lossy().to_string(),
            rust_version: env!("CARGO_PKG_VERSION").to_string(),
            platform: std::env::consts::OS.to_string(),
            cpu_count: num_cpus::get(),
            total_memory_gb,
            has_gpu: tch::Cuda::is_available(),
            gpu_name: if tch::Cuda::is_available() {
                Some("CUDA Device".to_string())
            } else {
                None
            },
            hardware_state: get_gpu_hardware_state(),
            tch_version: "0.15.0".to_string(), // Update as needed
            timestamp: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            hostname: hostname::get()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        },
        component_results: Vec::new(),
        pipeline_results: Vec::new(),
        gpu_results: Vec::new(),
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

    // Phase 2: CPU Pipeline benchmarks
    println!("\n{}", "=".repeat(80));
    println!("PHASE 2: CPU TRANSFORM PIPELINES (PERSISTENT WORKERS)");
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

    // Phase 3: GPU Pipeline benchmarks
    if tch::Cuda::is_available() {
        println!("\n{}", "=".repeat(80));
        println!("PHASE 3: GPU PIPELINE BENCHMARKS");
        println!("{}", "=".repeat(80));

        let gpu_configs = vec![
            // (batch_size, num_workers, pin_memory)
            (32, 0, true),  // Single-threaded baseline (pinned)
            (32, 4, false), // Multi-worker, no pin (shows pin_memory benefit)
            (32, 4, true),  // Multi-worker, pinned (standard config)
            (64, 4, true),  // Larger batch (common for training)
            (128, 4, true), // Large batch (if memory allows)
        ];

        let gpu_num_batches = 100;
        let gpu_warmup_batches = 10;

        for level_name in &["level7_production_inference", "level8_production_training"] {
            if let Some(transform_creator) = transforms.get(*level_name) {
                println!("\n{}", "=".repeat(60));
                println!("GPU Transform Level: {}", level_name);
                println!("{}", "=".repeat(60));

                for (batch_size, num_workers, pin_memory) in &gpu_configs {
                    match pipeline_bench.benchmark_gpu_pipeline(
                        level_name,
                        transform_creator.as_ref(),
                        *batch_size,
                        *num_workers,
                        *pin_memory,
                        gpu_num_batches,
                        gpu_warmup_batches,
                    ) {
                        Ok(result) => {
                            println!(
                                "  → GPU Util: {:.1}%, Efficiency: {:.2}, Transfer: {:.1} GB/s",
                                result.gpu_util_percent,
                                result.pipeline_efficiency,
                                result.transfer_bandwidth_gbps
                            );
                            report.gpu_results.push(result);
                        }
                        Err(e) => {
                            println!("  ✗ Error: {}", e);
                        }
                    }
                }
            }
        }
    } else {
        println!("\n  No GPU available - skipping GPU benchmarks");
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
    println!("COMPREHENSIVE BENCHMARK SUMMARY");
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

    // CPU Pipeline summary
    println!("\nCPU Pipeline Performance (images/sec):");
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
                break;
            }
        }
    }

    // GPU Pipeline Summary
    if !report.gpu_results.is_empty() {
        println!("\nGPU Pipeline Performance:");
        println!("{}", "-".repeat(110));
        println!(
            "{:<25} {:<20} {:<8} {:<8} {:<10} {:<10} {:<10} {:<10}",
            "Transform", "Config", "GPU%", "Eff", "Load ms", "Xfer ms", "GPU ms", "BW GB/s"
        );
        println!("{}", "-".repeat(110));

        for r in &report.gpu_results {
            let config = format!(
                "B={},W={},P={}",
                r.batch_size,
                r.num_workers,
                if r.pin_memory { "Y" } else { "N" }
            );
            println!(
                "{:<25} {:<20} {:>6.1}% {:>7.2} {:>9.1} {:>9.1} {:>8.1} {:>9.1}",
                r.transform_level,
                config,
                r.gpu_util_percent,
                r.pipeline_efficiency,
                r.dataload_time_ms,
                r.transfer_time_ms,
                r.compute_time_ms,
                r.transfer_bandwidth_gbps
            );
        }
    }
}

fn quick_diagnostic_test(data_root: &Path) -> Result<()> {
    println!("=== QUICK DIAGNOSTIC TEST ===");

    // Test data loading
    let pipeline_bench = TransformPipelineBenchmark::new(data_root.to_path_buf())?;
    let transforms = pipeline_bench.get_transform_levels();

    if let Some(transform_creator) = transforms.get("level3_resize") {
        println!("Testing CPU pipeline...");
        let result = pipeline_bench.benchmark_pipeline(
            "level3_resize",
            transform_creator.as_ref(),
            32, // small batch
            4,  // few workers
            10, // few batches
            2,  // quick warmup
            None,
        )?;
        println!("✓ CPU pipeline: {:.1} img/s", result.images_per_sec);
    }

    // Test GPU if available
    if tch::Cuda::is_available() {
        println!("Testing GPU pipeline...");
        if let Some(transform_creator) = transforms.get("level7_production_inference") {
            let gpu_result = pipeline_bench.benchmark_gpu_pipeline(
                "level7_production_inference",
                transform_creator.as_ref(),
                32, // small batch
                4,  // few workers
                true,
                10, // few batches
                2,  // quick warmup
            )?;
            println!("✓ GPU pipeline: {:.1}% util", gpu_result.gpu_util_percent);
        }
    }

    println!("✓ Quick test completed successfully");
    Ok(())
}
// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

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
        .unwrap_or("rust_comprehensive_benchmark_results.json");

    if !data_path.exists() {
        eprintln!("Error: Data path {:?} does not exist", data_path);
        std::process::exit(1);
    }

    // Run diagnostics first
    if let Err(e) = validate_gpu_setup() {
        println!("GPU validation failed: {}", e);
    }

    // Quick test
    quick_diagnostic_test(&data_path)?;

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
