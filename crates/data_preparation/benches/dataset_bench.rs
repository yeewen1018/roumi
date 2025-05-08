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
/// cargo bench --features boxed-iter
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
        let ds = make_dataset(size);

        group.bench_with_input(BenchmarkId::new("static", size), &ds, |b, ds| {
            b.iter(|| {
                let cnt = ds.iter().count();
                black_box(cnt);
            })
        });

        group.bench_with_input(BenchmarkId::new("boxed", size), &ds, |b, ds| {
            b.iter(|| {
                let cnt = ds.iter_boxed().count();
                black_box(cnt);
            })
        });
    }
    group.finish();
}

/// Measure iteration + a simple per-sample tensor operation.
fn bench_with_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("With Processing");
    for &size in &SIZES {
        let ds = make_dataset(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("static+map", size), &size, |b, &_s| {
            b.iter(|| {
                ds.iter()
                    .map(|r| {
                        let sample = r.unwrap();
                        let t = sample.get("input_ids").unwrap() + 1;
                        black_box(t)
                    })
                    .count()
            });
        });

        group.bench_with_input(BenchmarkId::new("boxed+map", size), &size, |b, &_s| {
            b.iter(|| {
                ds.iter_boxed()
                    .map(|r| {
                        let sample = r.unwrap();
                        let t = sample.get("input_ids").unwrap() + 1;
                        black_box(t)
                    })
                    .count()
            });
        });
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
