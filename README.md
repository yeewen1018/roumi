# roumi
Roumi is an open-source ecosystem of Rust libraries designed to address critical systems-level bottlenecks in large-scale machine learning.

**Note:** This is my active development fork of [`oumi-ai/roumi`](https://github.com/oumi-ai/roumi). 

---

## Project Highlights
This repository contains several key components, including:

* **A `data_preparation` crate** that provides a high-performance data loading framework. This component bypasses Python's GIL to achieve **up to 5.3x the throughput** and a **7x memory reduction** compared to PyTorch's standard DataLoader. This work was submitted to the NeurIPS ML for Systems Workshop, 2025.

* **A `grpo_rewards` crate** that calculates rewards for GRPO pipelines, achieving a **3x speedup** over Python baselines for CPU-bound tasks.

## Prerequisites

- **Rust:** [Install Rust](https://www.rust-lang.org/tools/install)
- **Python:** Version 3.10 or higher is recommended.
- **Maturin:** install via pip below

## Setting up a Python Virtual Environment 

(Optional, but Recommended)

```bash
python -m venv venv
venv\Scripts\activate # on Windows
source venv/bin/activate # on macOS/Linux
pip install -U pip
```

## Install Maturin
```bash
pip install maturin
```

## Build each crate for Python to use
```bash
cd crates/grpo_rewards
maturin develop
cd ../..
```

## Run the Python Example
```bash
cd ../../examples
python example_grpo_rewards.py
```

## Building and Running the Plugin
```bash
cd plugins/template
cargo build --release
cd ../../examples
python using_plugin_rewards.py
```

## Building and Running the Rust Code
```bash
cargo run
```

## Testing
```bash
cargo test --lib --bins --tests
```




