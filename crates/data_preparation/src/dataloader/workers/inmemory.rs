//! src/dataloader/workers/inmemory.rs
//!
//! Worker implementation for InMemoryDataset.
//!
//! Workers fetch transformed data (`Sample`s) on demand using indices
//! and collate them into mini-batches.
//!
//! # Architecture:
//! - Workers share the dataset via `Arc` for zero-copy access.
//! - Tasks are distributed via per-worker channels with work stealing
//! - Supports both fresh workers per epoch and persistent workers
//!
//! # Work stealing:
//! - Each worker has a dedicated primary queue for deterministic assignment
//! - Overflow tasks go to a shared steal queue when primary queues are full
//! - Workers check steal queue when their primary queue is empty
//! - RNG seedling uses original intended worker for determinism

use crate::collator::Collator;
use crate::dataset::InMemoryDataset;
use crate::minibatch::MiniBatch;
use crate::sample::Sample;
use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender, TrySendError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::pool::WorkerPool;
use super::STEAL_THRESHOLD_MS;
use crate::dataloader::common::thread::{init_worker_rng, WORKER_ID};
use crate::dataloader::config::DataLoaderConfig;

// Task types for in-memory worker communication.
#[derive(Debug)]
pub enum InMemoryWorkerTask {
    /// Process a batch of indices with deterministic assignment info
    Batch {
        indices: Vec<usize>,
        intended_worker: usize, // Original worker assignment for RNG determinism
        epoch: usize,           // For RNG seeding
    },
    /// Signal end of epoch
    EndEpoch,
    /// Set epoch for RNG initialization
    SetEpoch { epoch: usize, base_seed: u64 },
}

/// Manages workers for in-memory datasets.
pub(crate) struct InMemoryWorkerManager {
    pub(crate) worker_pool: WorkerPool<InMemoryWorkerTask, Result<MiniBatch>>,
    pub(crate) steal_queue: Option<(Sender<InMemoryWorkerTask>, Receiver<InMemoryWorkerTask>)>,
}

impl InMemoryWorkerManager {
    /// Creates a new manager with Arc-shared dataset access.
    ///
    /// Each worker receives batch indices and fetches samples on-demand.
    /// Returns error if persistent workers are requested but initialization fails.
    pub(crate) fn new<Raw, C>(
        num_workers: usize,
        dataset: Arc<InMemoryDataset<Raw>>,
        collator: C,
        config: &DataLoaderConfig,
    ) -> Result<Self>
    where
        Raw: Clone + Send + Sync + 'static,
        C: Collator + Clone + Send + Sync + 'static,
    {
        // Create steal queue if using workers
        let steal_queue = if num_workers > 0 {
            let capacity = num_workers * config.prefetch_factor * config.batch_size.unwrap() / 2;
            let (tx, rx) = bounded(capacity);
            Some((tx, rx))
        } else {
            None
        };

        if config.persistent_workers && num_workers > 0 {
            let pool = Self::create_persistent_pool(
                dataset,
                collator,
                num_workers,
                config,
                steal_queue
                    .as_ref()
                    .map(|(tx, rx)| (tx.clone(), rx.clone())),
            )?;
            Ok(Self {
                worker_pool: pool,
                steal_queue,
            })
        } else {
            // For fresh workers, create a minimal pool structure
            // The actual pool will be created in the iterator
            let dummy_pool = WorkerPool {
                workers: vec![],
                worker_task_txs: vec![],
                output_rx: crossbeam_channel::never(),
                shutdown: Arc::new(AtomicBool::new(false)),
            };

            Ok(Self {
                worker_pool: dummy_pool,
                steal_queue,
            })
        }
    }

    /// Creates a pool of persistent workers that survive across epochs.
    pub(crate) fn create_persistent_pool<Raw, C>(
        dataset: Arc<InMemoryDataset<Raw>>,
        collator: C,
        num_workers: usize,
        config: &DataLoaderConfig,
        steal_queue: Option<(Sender<InMemoryWorkerTask>, Receiver<InMemoryWorkerTask>)>,
    ) -> Result<WorkerPool<InMemoryWorkerTask, Result<MiniBatch>>>
    where
        Raw: Clone + Send + Sync + 'static,
        C: Collator + Clone + Send + Sync + 'static,
    {
        let buffer_size = config.prefetch_factor;
        let pin_memory = config.pin_memory;

        WorkerPool::new(
            num_workers,
            buffer_size,
            move |task_rx, output_tx, shutdown| {
                let worker_id = WORKER_ID.with(|id| *id.borrow());
                WORKER_ID.with(|id| *id.borrow_mut() = worker_id);

                let steal_rx = steal_queue.as_ref().map(|(_, rx)| rx.clone());
                let mut current_base_seed: Option<u64> = None;

                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }

                    // Try primary queue with timeout, then try steal queue
                    let task = match task_rx.recv_timeout(Duration::from_millis(STEAL_THRESHOLD_MS))
                    {
                        Ok(task) => task,
                        Err(RecvTimeoutError::Timeout) => {
                            // Primary empty, try steal queue
                            if let Some(ref steal_rx) = steal_rx {
                                match steal_rx.try_recv() {
                                    Ok(task) => task,
                                    Err(_) => continue, // No work, loop back
                                }
                            } else {
                                continue; // No steal queue, loop back
                            }
                        }
                        Err(RecvTimeoutError::Disconnected) => break,
                    };

                    match task {
                        InMemoryWorkerTask::SetEpoch { epoch, base_seed } => {
                            current_base_seed = Some(base_seed);
                            init_worker_rng(worker_id, epoch, base_seed);
                        }
                        InMemoryWorkerTask::Batch {
                            indices,
                            intended_worker,
                            epoch,
                        } => {
                            // Use intended worker for RNG seeding
                            if let Some(seed) = current_base_seed {
                                init_worker_rng(intended_worker, epoch, seed);
                            }

                            let result =
                                Self::process_batch_lazy(&dataset, &indices, &collator, pin_memory)
                                    .with_context(|| {
                                        format!("Persistent worker {} failed", worker_id)
                                    });

                            if output_tx.send(result).is_err() {
                                break;
                            }
                        }
                        InMemoryWorkerTask::EndEpoch => {
                            let _ = output_tx.send(Err(anyhow!("WORKER_EPOCH_DONE")));
                        }
                    }
                }
            },
        )
    }

    /// Process a batch by fetching samples on-demand using O(1) index access.
    ///
    /// This avoids pre-caching all samples in each worker. Called by both
    /// fresh and persistent workers.
    pub(crate) fn process_batch_lazy<Raw, C>(
        dataset: &InMemoryDataset<Raw>,
        indices: &[usize],
        collator: &C,
        pin_memory: bool,
    ) -> Result<MiniBatch>
    where
        Raw: Clone + Send + Sync + 'static,
        C: Collator,
    {
        let samples_result: Result<Vec<Sample>> = indices
            .iter()
            .map(|&index| {
                dataset.get_sample(index).with_context(|| {
                    format!(
                        "Failed to load sample at index {} (dataset size: {})",
                        index,
                        dataset.len()
                    )
                })
            })
            .collect();

        let samples = samples_result?;
        let batch = collator
            .collate(&samples)
            .with_context(|| format!("Failed to collate batch of {} samples", samples.len()))?;

        Ok(if pin_memory {
            batch.pin_memory()
        } else {
            batch
        })
    }

    /// Sends a batch to specific worker with work stealing fallback
    pub(crate) fn send_task_to_worker(
        &self,
        worker_id: usize,
        indices: Vec<usize>,
        epoch: usize,
    ) -> Result<()> {
        let task = if indices.is_empty() {
            InMemoryWorkerTask::EndEpoch
        } else {
            InMemoryWorkerTask::Batch {
                indices,
                intended_worker: worker_id,
                epoch,
            }
        };

        let worker_tx = &self.worker_pool.worker_task_txs[worker_id];

        // Try primary queue first (non-blocking)
        match worker_tx.try_send(task) {
            Ok(_) => {
                // Success - sent to intended worker
                Ok(())
            }
            Err(TrySendError::Full(task)) => {
                // Primary queue full, try steal queue if available
                if let Some((ref steal_tx, _)) = self.steal_queue {
                    match steal_tx.try_send(task) {
                        Ok(_) => {
                            // Successfully send task to intended worker
                            Ok(())
                        }
                        Err(TrySendError::Full(task)) => {
                            // Both queues full, fall back to blocking send
                            worker_tx
                                .send(task)
                                .map_err(|_| anyhow!("Worker {} channel closed", worker_id))
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            Err(anyhow!("Steal queue disconnected"))
                        }
                    }
                } else {
                    // No steal queue, fall back to blocking send
                    worker_tx
                        .send(task)
                        .map_err(|_| anyhow!("Worker {} channel closed", worker_id))
                }
            }
            Err(TrySendError::Disconnected(_)) => {
                Err(anyhow!("Worker {} has disconnected", worker_id))
            }
        }
    }

    /// Receives a processed MiniBatch from the worker pool.
    /// Blocks until a result is available or timeout occurs.
    pub(crate) fn receive_task_result(&self, timeout: Duration) -> Result<Result<MiniBatch>> {
        self.worker_pool
            .output_rx
            .recv_timeout(timeout)
            .map_err(|e| match e {
                RecvTimeoutError::Timeout => anyhow!(
                    "Worker timeout after {:?} - possible deadlock or slow data loading",
                    timeout
                ),
                RecvTimeoutError::Disconnected => {
                    anyhow!("Worker channel disconnected - workers may have crashed")
                }
            })
    }

    /// Send epoch information to all workers
    pub(crate) fn set_epoch(&self, epoch: usize, base_seed: u64) -> Result<()> {
        for tx in &self.worker_pool.worker_task_txs {
            tx.send(InMemoryWorkerTask::SetEpoch { epoch, base_seed })
                .map_err(|_| anyhow!("Failed to send epoch info to workers"))?;
        }
        Ok(())
    }

    pub(crate) fn wait_for_completion(&self) -> Result<()> {
        self.worker_pool.wait_for_completion()
    }
}
