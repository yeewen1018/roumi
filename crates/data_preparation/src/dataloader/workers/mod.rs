//! Worker management for parallel data loading.
//!
//! This module provides the abstractions for multi-threaded data loading:
//! - `pool`: Worker pool implementation
//! - `inmemory`: Workers for indexed datasets with random access
//! - `iterable`: Workers for streaming datasets
//!
//! Workers communicate through channels and share a common control protocol
//! defined by the `WorkerControl` enum.

pub(crate) mod inmemory;
pub(crate) mod iterable;
pub(crate) mod pool;

/// Control messages for worker coordination.
///
/// Used by persistent workers to synchronize epoch boundaries.
#[derive(Debug)]
pub enum WorkerControl {
    /// Signal workers to begin processing a new epoch
    StartEpoch { epoch: usize, base_seed: u64 },
}

// Work stealing threshold - how long workers wait before checking steal queue
pub(crate) const STEAL_THRESHOLD_MS: u64 = 10;

/// Timeout for draining remaining outputs during worker pool shutdown (milliseconds)
pub(crate) const COMPLETION_TIMEOUT_MS: u64 = 100;
