use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use data_preparation::{Dataset, InMemoryDataset, Sample};
use tch::{Device, Kind, Tensor};

/// Benchmarks for `InMemoryDataset` iterator performance.
///
/// This measures:
/// 1. Iteration overhead: the cost of `.iter()` (static) vs. `.iter_boxed()` (boxed dispatch)
/// 2. With processing: the combined cost of iteration + lightweight tensor operation
///
/// To run these, use:
/// ```bash
/// cargo bench
/// ```

/// All tests sweep across dataset sizes from 1K to 1M samples.
const SIZES: [usize; 4] = [1_000, 10_000, 100_000, 1_000_000];

/// Helper function to build an `InMemoryDataset` of the given size.
fn make_dataset(size: usize) -> InMemoryDataset {
    let samples = (0..size)
        .map(|i| {
            Sample::from_single(
                "input_ids",
                Tensor::from_slice(&[i as i64]).to_kind(Kind::Int64),
            )
            .with_feature(
                "attention_mask",
                Tensor::ones(&[1], (Kind::Float, Device::Cpu)),
            )
        })
        .collect();
    InMemoryDataset::new(samples)
}

/// Measure pure iteration overhead
fn bench_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("Iteration Overhead");

    for &size in &SIZES {
        let dataset = make_dataset(size);

        group.bench_with_input(
            BenchmarkId::new("static", size),
            &dataset,
            |bench, dataset| {
                bench.iter(|| {
                    let count = dataset.iter().count();
                    black_box(count);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("boxed", size),
            &dataset,
            |bench, dataset| {
                bench.iter(|| {
                    let count = dataset.iter_boxed().count();
                    black_box(count);
                })
            },
        );
    }
    group.finish();
}

/// Measure iteration + a simple per-sample tensor operation.
fn bench_with_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("With Processing");
    for &size in &SIZES {
        let dataset = make_dataset(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("static+map", size),
            &size,
            |bench, &_dataset_size| {
                bench.iter(|| {
                    dataset
                        .iter()
                        .map(|sample_result| {
                            let sample = sample_result.unwrap();
                            let processed_sample = sample.get("input_ids").unwrap() + 1;
                            black_box(processed_sample)
                        })
                        .count()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("boxed+map", size),
            &size,
            |bench, &_dataset_size| {
                bench.iter(|| {
                    dataset
                        .iter_boxed()
                        .map(|sample_result| {
                            let sample = sample_result.unwrap();
                            let processed_sample = sample.get("input_ids").unwrap() + 1;
                            black_box(processed_sample)
                        })
                        .count()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5))
        .sample_size(50);
    targets = bench_iteration, bench_with_processing
);
criterion_main!(benches);
