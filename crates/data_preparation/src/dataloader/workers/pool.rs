//! Worker pool implementation for parallel data loading.
//!
//! Provides a reusable thread pool that manages worker lifecycle, task distribution,
//! and result collection. This is used by both in-memory and iterable dataset
//! workers to parallelize data loading and transformation.
//!
//! # Key features
//! - Bounded channels prevent memory bloat
//! - Graceful shutdown on drop
//! - Thread-local worker IDs for debugging and sharding
//! - Generic over task and output types for flexibility
//!
//! # Distribution Modes
//!
//! Supports two distribution strategies, both ensuring deterministic task routing
//! for reproducible results with random transforms:
//!
//! - **Shared channel** (`new`): Workers pull from a single queue. Used when
//!   determinism is handled at a higher level (e.g., pre-sharded data sources).
//!
//! - **Per-worker channels** (`new_deterministic`): Tasks are routed to specific
//!   workers. Used when the pool must enforce deterministic assignment.

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use super::COMPLETION_TIMEOUT_MS;
use crate::dataloader::common::thread::WORKER_ID;

/// Thread pool for parallel data loading.
///
/// Manages worker lifecycle and communication through bounded channels:
/// - Task channel(s): Main thread -> Workers (work distribution)
/// - Output channel: Workers -> Main thread (result collection)
/// - Shutdown flag: Enables graceful termination
///
/// # Type Parameters
/// - `Task`: Work items sent to workers
/// - `Output`: Results returned from workers
pub(crate) struct WorkerPool<Task, Output> {
    pub(crate) workers: Vec<thread::JoinHandle<()>>,
    pub(crate) worker_task_txs: Vec<Sender<Task>>,
    pub(crate) output_rx: Receiver<Output>,
    pub(crate) shutdown: Arc<AtomicBool>,
}

impl<Task, Output> WorkerPool<Task, Output>
where
    Task: Send + 'static,
    Output: Send + 'static,
{
    /// Creates a new worker pool with per-worker channels.
    ///
    /// Tasks are routed to specific workers by the main thread. Ensures
    /// consistent worker assignment for reproducible random transforms.
    pub(crate) fn new<F>(num_workers: usize, buffer_size: usize, worker_fn: F) -> Result<Self>
    where
        F: Fn(Receiver<Task>, Sender<Output>, Arc<AtomicBool>) + Send + Sync + 'static,
    {
        // Validation
        if num_workers == 0 {
            return Err(anyhow!(
                "Cannot create WorkerPool with 0 workers. \
                Either set num_workers > 0 or use single-threaded mode."
            ));
        }

        if buffer_size == 0 {
            return Err(anyhow!(
                "Cannot create WorkerPool with buffer_size 0. \
                Buffer size must be > 0 to prevent deadlocks."
            ));
        }

        // Create per-worker channels
        let mut worker_task_txs = Vec::with_capacity(num_workers);
        let mut task_receivers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let (tx, rx) = bounded(buffer_size);
            worker_task_txs.push(tx);
            task_receivers.push(rx);
        }

        // Create shared output channel
        let output_buffer_size = buffer_size * num_workers;
        let (output_tx, output_rx) = bounded(output_buffer_size);

        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_fn = Arc::new(worker_fn);
        let mut workers = Vec::with_capacity(num_workers);

        // Spawn workers
        for (worker_id, task_rx) in task_receivers.into_iter().enumerate() {
            let output_tx = output_tx.clone();
            let shutdown_clone = shutdown.clone();
            let worker_fn_clone = worker_fn.clone();

            let handle = thread::Builder::new()
                .name(format!("dataloader-worker-{}", worker_id))
                .spawn(move || {
                    WORKER_ID.with(|id| *id.borrow_mut() = worker_id);
                    worker_fn_clone(task_rx, output_tx, shutdown_clone);
                })
                .with_context(|| format!("Failed to spawn worker thread {}", worker_id))?;

            workers.push(handle);
        }

        Ok(Self {
            workers,
            worker_task_txs,
            output_rx,
            shutdown,
        })
    }

    /// Wait for all workers to finish processing their current tasks
    pub(crate) fn wait_for_completion(&self) -> Result<()> {
        // Close all worker channels to signal completion
        for tx in &self.worker_task_txs {
            drop(tx.clone());
        }

        // Drain any remaining outputs
        while let Ok(_) = self
            .output_rx
            .recv_timeout(std::time::Duration::from_millis(COMPLETION_TIMEOUT_MS))
        {
            // Discard remaining outputs
        }
        Ok(())
    }
}

impl<Task, Output> Drop for WorkerPool<Task, Output> {
    fn drop(&mut self) {
        // Signal shutdown to all workers
        self.shutdown.store(true, Ordering::Relaxed);

        // Also drop per-worker channels if they exist
        self.worker_task_txs.clear();

        // Wait for workers to finish
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
