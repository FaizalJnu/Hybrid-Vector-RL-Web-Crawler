# Hybrid Vector-RL Focused Web Crawler

This repository contains the implementation of a **Topic-Agnostic Focused Web Crawler** utilizing a Hybrid Vector-Reinforcement Learning architecture. Designed for edge deployment (e.g., NVIDIA Jetson), this system leverages **Offline Reinforcement Learning** to navigate web hierarchies using geometric relevance gradients rather than keyword matching.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Workflow Overview](#workflow-overview)
- [Reproduction Steps](#reproduction-steps)
  - [1. Data Collection](#1-data-collection)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Training (Offline RL)](#3-training-offline-rl)
  - [4. Deployment & Testing](#4-deployment--testing)
  - [5. Benchmarking & Results](#5-benchmarking--results)

## 🛠️ Prerequisites
* **Python 3.8+**
* **CUDA-capable GPU** (Recommended for training) or NVIDIA Jetson (for inference)
* **Groq API Key** (For LLM-based trajectory evaluation)

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/vector-rl-crawler.git](https://github.com/yourusername/vector-rl-crawler.git)
    cd vector-rl-crawler
    ```

2.  Install dependencies:
    ```bash
    pip install torch torchvision numpy pandas matplotlib scikit-learn
    pip install sentence-transformers beautifulsoup4 requests groq
    ```

---

## Workflow Overview

The experiment follows a strict pipeline: **Collection $\to$ Vectorization $\to$ Training $\to$ Deployment $\to$ Evaluation**.

### 1. Data Collection
We first build a dataset of navigation transitions from a source domain (e.g., EuroLeague Basketball) to train the agent on general navigation logic.

* **Command:** `python datacollector.py`
* **Output:** `raw_crawl_data.jsonl`
* **Details:** Crawls approximately **2,000 pages**, extracting context text and roughly **30-40 links per page**.

### 2. Preprocessing
Convert the raw textual logs into high-dimensional vector tensors using Sentence-BERT. This prepares the "State-Action" pairs for the Q-Network.

* **Command:** `python preprocess.py`
* **Input:** `raw_crawl_data.jsonl`
* **Output:** `training_tensors.pt` (Serialized PyTorch tensors)
* **Process:** Encodes page context ($S$) and link text ($A$) into 384-dimensional embeddings.

### 3. Training (Offline RL)
Train the Deep Q-Network (DQN) using the offline dataset. The agent learns to predict the long-term reward (Relevance) of clicking a link based on the page context.

* **Command:** `python vectortrain.py`
* **Configuration:** Training runs for **200,000 steps** using Hindsight Experience Replay (HER).
* **Output:** `checkpoint_advanced.pth` (The trained "Brain").

### 4. Deployment & Testing
Deploy the trained agent in a live, unseen environment (e.g., NYU Engineering Domain) to test Zero-Shot Generalization. You must run this step **twice** to generate comparison data: once for the RL Agent and once for the Naive Baseline.

**Run A: Reinforcement Learning Mode**
1.  Open `deployment.py`.
2.  Set `USE_RL = True` in the configuration section.
3.  Run:
    ```bash
    python deployment.py
    ```
4.  **Output:** `crawl_results_RL.jsonl` and `frontier_snapshot.json`

**Run B: Naive (Greedy) Mode**
1.  Open `deployment.py`.
2.  Set `USE_RL = False`.
3.  Run:
    ```bash
    python deployment.py
    ```
4.  **Output:** `crawl_results_Naive.jsonl`

### 5. Benchmarking & Results
Finally, generate the thesis metrics (Harvest Rate, Latency, Trajectory Gain) and visualization plots.

* **Command:** `python v2_bench_metrics.py`
* **Inputs:** Reads the `.jsonl` logs generated in Step 4.
* **Outputs:** * Console Report (Precision, Efficiency, Latency)
    * `results_plot_RL.png` (Trajectory Graph)
    * `results_plot_Naive.png` (Trajectory Graph)

---

## Expected Results
Upon completion, the benchmarking script should reveal:
* **Latency:** RL Agent is comparable to or faster than the Naive baseline (~930ms).
* **Trajectory:** RL Agent shows a positive learning curve (recovering from low-reward dips), indicating successful "Tunneling" behavior.
* **Harvest Rate:** RL Agent may have lower raw precision (due to exploration) but higher peak discovery potential.
