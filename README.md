# Adaptive Splay Tree Optimization Using Monte Carlo Search and Q-Learning with Caching Mechanisms

## Overview

This repository presents an adaptive data structure named **MCSTS-Q Splay Tree**, which integrates **Monte Carlo Search**, **Q-Learning**, and **Caching Mechanisms** to optimize splay tree performance. The primary goal of this project is to minimize the costly overhead of unnecessary rotations in splay trees, especially in scenarios where non-uniform access patterns (like temporal or skewed distributions) occur.

### Key Features:

- **Monte Carlo Search (MCS)** for selective splay decisions to minimize future access costs probabilistically.
- **Q-Learning** to adapt and refine splay and caching strategies based on real-time data access patterns.
- **Caching Mechanism** to cache frequently accessed nodes, thus avoiding the need for repeated costly rotations.
- **Ablation Studies**: All techniques (MCS, Q-Learning, and Caching) are modular and can be enabled/disabled for ablation experiments, allowing for performance evaluations of individual techniques and their combinations.

### Repository Structure

```
AdaptiveSplayTree/
├── README.md
├── requirements.txt
├── splay_tree.py
├── mcsts_q_splay_tree.py
├── run_experiments.py
├── data/
└── results/
```

- **README.md**: This file, providing an overview of the project, how to use it, and other relevant details.
- **requirements.txt**: Lists the dependencies required to run the project.
- **splay_tree.py**: Implementation of the traditional splay tree.
- **mcsts_q_splay_tree.py**: Implementation of the MCSTS-Q Splay Tree, with Monte Carlo Search, Q-Learning, and Caching.
- **run_experiments.py**: Script for running experiments across different configurations, including ablation tests.
- **data/**: Directory to store data for experimentation.
- **results/**: Directory to store experimental results and logs.

### Techniques Used

- **Splay Tree**: A self-adjusting binary search tree that brings frequently accessed nodes closer to the root.
- **Monte Carlo Search**: A probabilistic approach used to evaluate possible splay operations to minimize future costs.
- **Q-Learning**: A reinforcement learning technique that helps adapt the tree to real-time access patterns, learning optimal splay and caching policies.
- **Caching Mechanism**: Uses Q-Learning to decide whether to cache nodes, reducing the frequency of splay operations.

### Ablation Studies

The implementation allows enabling/disabling the following components for ablation studies:

- **Monte Carlo Search (MCS)**: Enable or disable probabilistic optimization of splay operations.
- **Q-Learning**: Enable or disable the Q-Learning agent that refines splay and caching strategies.
- **Caching Mechanism**: Enable or disable the caching of frequently accessed nodes.

### Access Patterns for Testing

The following access patterns are supported for evaluation:

- **Uniform**: Nodes are accessed with equal probability.
- **Skewed**: Some nodes are accessed more frequently, following a Zipfian distribution.
- **Temporal**: Recently accessed nodes are more likely to be accessed again, simulating real-world usage.
- **Cluster-based, hierarchical, random walk, and bursty patterns**: Additional real-world access scenarios are available for more comprehensive testing.

### Performance Metrics

- **Number of Rotations**: Measures the overhead caused by tree restructuring.
- **Average Depths**: Measures the average node depth before and after splay operations.
- **Cache Hit Rates**: Measures the effectiveness of the caching mechanism.
- **Reward Convergence**: Monitors the performance of the Q-Learning agent over time.

## Getting Started

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/AdaptiveSplayTree.git
   cd AdaptiveSplayTree
   ```

2. **Create a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments

To run all experiments, use the command below:

```bash
python run_experiments.py
```

This command will run the experiments with all configurations, including ablation settings, and produce visualizations of the results.

## Hyperparameter Tuning

To find the optimal cache size and Q-Learning parameters for different access patterns, perform a hyperparameter search using:

```bash
python run_experiments.py --hyperparameter_search
```

## Results Visualization

- The experimental results are saved in the **results/** directory.
- Use **Plotly** to visualize metrics like rotations over time, average depths, cache hit rates, and Q-Learning rewards.

## Contributions

Contributions are welcome! If you'd like to improve this repository, please fork the repository and submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See **LICENSE** for more information.

## Contact

If you have questions or suggestions, feel free to reach out at [your email].

---

This repository is built to help bridge the gap between traditional self-adjusting data structures and modern reinforcement learning techniques, aiming to provide both educational value and practical performance improvements in adaptive tree structures.

