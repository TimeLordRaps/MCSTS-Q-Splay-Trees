# run_experiments.py

import numpy as np
from tqdm import tqdm
from langchain_ollama.embeddings import OllamaEmbeddings
from mcsts_q_splay_tree import MCSTSQSplayTree
from splay_tree import SplayTree
import pandas as pd
import pickle
import os
import json
import logging
import gc
import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns
import scipy.stats as stats
import plotly.graph_objects as go
import psutil
import cProfile
import pstats
from io import StringIO
from datasets import load_dataset

# ==========================
# 1. Imports and Logging Configuration
# ==========================

def setup_logging(log_file: str):
    """
    Sets up logging to both console and file with detailed formatting.

    Parameters:
        log_file (str): Path to the log file.
    """
    logger = logging.getLogger('ExperimentLogger')
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Formatter for detailed logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler for INFO level and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # File handler for DEBUG level and above
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Avoid duplicate logs
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

# Initialize logger
log_file_path = 'results/logs/experiment.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logger = setup_logging(log_file_path)

# ==========================
# 2. Utility Functions
# ==========================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Computes the cosine similarity between two vectors.

    Parameters:
        vec1 (List[float]): First vector.
        vec2 (List[float]): Second vector.

    Returns:
        float: Cosine similarity.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def save_splay_tree(tree: MCSTSQSplayTree, filepath: str):
    """
    Saves the splay tree to a file using pickle.

    Parameters:
        tree (MCSTSQSplayTree): The splay tree to save.
        filepath (str): The path to the file where the tree will be saved.
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(tree, f)
        logger.info(f"Splay tree saved successfully to '{filepath}'.")
    except Exception as e:
        logger.error(f"Failed to save splay tree to '{filepath}': {e}")

def load_splay_tree(filepath: str) -> MCSTSQSplayTree:
    """
    Loads the splay tree from a pickle file.

    Parameters:
        filepath (str): The path to the pickle file.

    Returns:
        MCSTSQSplayTree: The loaded splay tree.
    """
    try:
        with open(filepath, 'rb') as f:
            tree = pickle.load(f)
        logger.info(f"Splay tree loaded successfully from '{filepath}'.")
        return tree
    except Exception as e:
        logger.error(f"Failed to load splay tree from '{filepath}': {e}")
        raise e

def save_results(data, filepath: str):
    """
    Saves data to a JSON file.

    Parameters:
        data (dict): The data to save.
        filepath (str): The path to the JSON file.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Results saved successfully to '{filepath}'.")
    except Exception as e:
        logger.error(f"Failed to save results to '{filepath}': {e}")

def load_dataset_from_hf(dataset_name: str, configs: List[str], split: str = 'test') -> pd.DataFrame:
    """
    Loads multiple configurations of a dataset from Hugging Face datasets library
    and concatenates them into a single DataFrame.

    Parameters:
        dataset_name (str): The name of the dataset on Hugging Face.
        configs (List[str]): List of configuration names to load.
        split (str): The data split to load ('test' in this case).

    Returns:
        pd.DataFrame: The concatenated dataset.
    """
    try:
        logger.info(f"Loading dataset '{dataset_name}' from Hugging Face with configs: {configs} and split: '{split}'")
        df_list = []
        for config in configs:
            logger.info(f"Loading config '{config}'...")
            dataset = load_dataset(dataset_name, f"{config}_e", split=split)
            temp_df = pd.DataFrame(dataset)
            temp_df['text'] = temp_df['input'] + '\n' + temp_df['context']
            # Ensure the DataFrame has 'id' and 'text' columns
            if 'id' not in temp_df.columns:
                temp_df['id'] = temp_df.index
            if 'text' not in temp_df.columns:
                raise ValueError(f"Config '{config}' does not contain a 'text' column.")
            df_list.append(temp_df[['id', 'text']])
        concatenated_df = pd.concat(df_list, ignore_index=False)
        logger.info(f"Total records after concatenation: {concatenated_df.shape[0]}")
        return concatenated_df
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        raise e

def embed_and_store(df: pd.DataFrame, embed: OllamaEmbeddings, tree: MCSTSQSplayTree):
    """
    Embeds each text sample and stores the embedding in the splay tree.
    
    Parameters:
        df (pd.DataFrame): The dataset containing text samples.
        embed (OllamaEmbeddings): The embedding instance.
        tree (MCSTSQSplayTree): The splay tree to store embeddings.
    """
    logger.info("Starting embedding and storage of text samples.")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Embedding and Storing"):
        text_id = index
        text = str(row.get('text', ''))  # Use 'text' column
        if not text:
            logger.warning(f"No text found for ID {text_id}, skipping.")
            continue
        # print(text)
        try:
            vector = embed.embed_query(str(text))
            # Store the embedding in the splay tree with text_id as the key
            tree.insert(key=text_id, value=vector)
            logger.debug(f"Embedded and stored text ID {text_id}.")
        except Exception as e:
            logger.error(f"Embedding failed for ID {text_id}: {e}")
    logger.info("Completed embedding and storage of text samples.")
# ==========================
# 3. Embedding and Splay Tree Integration
# ==========================

def initialize_embeddings(model_name: str) -> OllamaEmbeddings:
    """
    Initializes the OllamaEmbeddings with the specified model.

    Parameters:
        model_name (str): The name of the Ollama model to use.

    Returns:
        OllamaEmbeddings: An instance of OllamaEmbeddings.
    """
    try:
        embed = OllamaEmbeddings(
            model=model_name,
            base_url='http://localhost:11434'
        )
        logger.info(f"OllamaEmbeddings initialized with model '{model_name}'.")
        return embed
    except Exception as e:
        logger.error(f"Failed to initialize OllamaEmbeddings with model '{model_name}': {e}")
        raise e

def initialize_splay_tree(config: dict) -> MCSTSQSplayTree:
    """
    Initializes the MCSTSQSplayTree with the given configuration.

    Parameters:
        config (dict): Configuration parameters for the splay tree.

    Returns:
        MCSTSQSplayTree: An initialized splay tree instance.
    """
    try:
        tree = MCSTSQSplayTree(
            use_mcs=config.get('use_mcs', True),
            use_qlearning=config.get('use_qlearning', True),
            use_cache=config.get('use_cache', True),
            cache_size=config.get('cache_size', 256),
            q_params={
                'alpha': config.get('alpha', 0.1),
                'gamma': config.get('gamma', 0.9),
                'epsilon': config.get('epsilon', 0.1)
            },
            mcs_params={'simulations': config.get('mcs_sims', 100)}
        )
        logger.info("Splay tree initialized with provided configuration.")
        return tree
    except Exception as e:
        logger.error(f"Failed to initialize splay tree: {e}")
        raise e

# ==========================
# 4. Semantic Comparison Functions
# ==========================

def semantic_search(tree: MCSTSQSplayTree, embed: OllamaEmbeddings, query: str, top_n: int =5) -> List[Tuple[str, float]]:
    """
    Performs a semantic search to find the top N most similar texts to the query.

    Parameters:
        tree (MCSTSQSplayTree): The splay tree containing embeddings.
        embed (OllamaEmbeddings): The embedding instance.
        query (str): The query text.
        top_n (int): Number of top similar texts to return.

    Returns:
        List[Tuple[str, float]]: List of tuples containing text IDs and their similarity scores.
    """
    try:
        query_vector = embed.embed_query(query)
        logger.debug("Query text embedded successfully.")
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return []

    similarities = []

    # Traverse the splay tree and compute similarity
    def traverse(node):
        if node is not None:
            traverse(node.left)
            if isinstance(node.value, list) and len(node.value) >0:
                sim = cosine_similarity(query_vector, node.value)
                similarities.append((node.key, sim))
            traverse(node.right)

    traverse(tree.root)

    # Sort based on similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top N
    logger.info(f"Semantic search completed. Top {top_n} similar texts retrieved.")
    return similarities[:top_n]

# ==========================
# 5. Experimentation and Ablation Studies
# ==========================

def perform_ablation_study(df: pd.DataFrame, embed: OllamaEmbeddings, config: dict, ablation_configs: List[dict], tree_save_path: str):
    """
    Performs ablation studies by varying configuration parameters.

    Parameters:
        df (pd.DataFrame): The dataset containing text samples.
        embed (OllamaEmbeddings): The embedding instance.
        config (dict): Base configuration for the splay tree.
        ablation_configs (List[dict]): List of configuration overrides for ablations.
        tree_save_path (str): Path to save the splay tree.

    Returns:
        dict: Results of the ablation studies.
    """
    logger.info("Starting ablation studies.")
    ablation_results = {}
    for idx, ablation in enumerate(ablation_configs):
        logger.info(f"Running ablation study {idx+1}/{len(ablation_configs)} with configuration: {ablation}")
        # Merge base config with ablation config
        current_config = config.copy()
        current_config.update(ablation)
        
        # Initialize splay tree with current configuration
        tree = initialize_splay_tree(current_config)
        
        # Embed and store data
        embed_and_store(df, embed, tree)
        
        # Save the splay tree
        ablation_tree_path = f"{tree_save_path}_ablation_{idx+1}.pkl"
        save_splay_tree(tree, ablation_tree_path)
        
        # Evaluate performance metrics
        depths = [tree._get_depth(node) for node in traverse_tree(tree.root)]
        avg_depth = np.mean(depths) if depths else 0
        total_rotations = tree.total_rotations
        cache_stats = tree.get_cache_stats() if hasattr(tree, 'get_cache_stats') else {}
        cache_hit_rate = cache_stats.get('hit_rate', 0)
        
        ablation_results[f"ablation_{idx+1}"] = {
            'configuration': current_config,
            'avg_depth': avg_depth,
            'total_rotations': total_rotations,
            'cache_hit_rate': cache_hit_rate
        }
        logger.info(f"Ablation {idx+1} results: Avg Depth = {avg_depth:.4f}, "
                    f"Total Rotations = {total_rotations}, Cache Hit Rate = {cache_hit_rate:.4f}")
        
        # Clean up
        del tree
        gc.collect()
    
    logger.info("Ablation studies completed.")
    return ablation_results

def traverse_tree(node):
    """
    Generator to traverse the splay tree in-order.

    Parameters:
        node (Node): The current node.

    Yields:
        Node: The next node in the traversal.
    """
    if node is not None:
        yield from traverse_tree(node.left)
        yield node
        yield from traverse_tree(node.right)

# ==========================
# 6. Comprehensive Analysis Functions
# ==========================

def parameter_sensitivity_analysis(results: dict) -> dict:
    """
    Analyzes the sensitivity of performance metrics to different parameters.

    Parameters:
        results (dict): Results from ablation studies or hyperparameter tuning.

    Returns:
        dict: Sensitivity analysis results.
    """
    logger.info("Performing parameter sensitivity analysis.")
    sensitivity = {}
    for ablation, metrics in results.items():
        for param, value in metrics['configuration'].items():
            if param not in sensitivity:
                sensitivity[param] = {}
            if value not in sensitivity[param]:
                sensitivity[param][value] = {'avg_depths': [], 'total_rotations': [], 'cache_hit_rates': []}
            sensitivity[param][value]['avg_depths'].append(metrics['avg_depth'])
            sensitivity[param][value]['total_rotations'].append(metrics['total_rotations'])
            sensitivity[param][value]['cache_hit_rates'].append(metrics['cache_hit_rate'])

    # Calculate statistics
    sensitivity_stats = {}
    for param, values in sensitivity.items():
        sensitivity_stats[param] = {}
        for value, metrics in values.items():
            sensitivity_stats[param][value] = {
                'mean_avg_depth': np.mean(metrics['avg_depths']),
                'std_avg_depth': np.std(metrics['avg_depths']),
                'mean_total_rotations': np.mean(metrics['total_rotations']),
                'std_total_rotations': np.std(metrics['total_rotations']),
                'mean_cache_hit_rate': np.mean(metrics['cache_hit_rates']),
                'std_cache_hit_rate': np.std(metrics['cache_hit_rates'])
            }
    logger.info("Parameter sensitivity analysis completed.")
    return sensitivity_stats

def correlation_analysis(ablation_results: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Pearson and Spearman correlation analysis between parameters and performance metrics.

    Parameters:
        ablation_results (dict): Results from ablation studies.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Pearson and Spearman correlation matrices.
    """
    logger.info("Performing correlation analysis.")
    data = []
    for ablation, metrics in ablation_results.items():
        entry = metrics['configuration'].copy()
        entry.update({
            'avg_depth': metrics['avg_depth'],
            'total_rotations': metrics['total_rotations'],
            'cache_hit_rate': metrics['cache_hit_rate']
        })
        data.append(entry)
    df = pd.DataFrame(data)
    pearson_corr = df.corr(method='pearson')
    spearman_corr = df.corr(method='spearman')
    
    # Save correlation matrices
    pearson_corr.to_csv('results/correlation_pearson.csv')
    spearman_corr.to_csv('results/correlation_spearman.csv')
    logger.info("Correlation analysis completed and saved.")
    return pearson_corr, spearman_corr

def runtime_performance_measurement(tree_class, access_pattern: List[str]) -> float:
    """
    Measures the runtime performance of tree operations.

    Parameters:
        tree_class (class): The tree class to instantiate.
        access_pattern (List[str]): The sequence of keys to access.

    Returns:
        float: Total runtime in seconds.
    """
    import time
    logger.info("Measuring runtime performance.")
    start_time = time.time()
    tree = tree_class(
        use_mcs=True,
        use_qlearning=True,
        use_cache=True,
        cache_size=256,
        q_params={
            'alpha': 0.1,
            'gamma': 0.9,
            'epsilon': 0.1
        },
        mcs_params={'simulations': 100}
    )
    for key in tqdm(access_pattern, desc="Runtime Measurement"):
        tree.access(key)
    end_time = time.time()
    runtime = end_time - start_time
    logger.info(f"Runtime Performance: {runtime:.4f} seconds.")
    return runtime

def memory_usage_analysis(tree_class, access_pattern: List[str]) -> float:
    """
    Analyzes memory usage during tree operations.

    Parameters:
        tree_class (class): The tree class to instantiate.
        access_pattern (List[str]): The sequence of keys to access.

    Returns:
        float: Memory used in MB.
    """
    logger.info("Analyzing memory usage.")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # in MB
    tree = tree_class(
        use_mcs=True,
        use_qlearning=True,
        use_cache=True,
        cache_size=256,
        q_params={
            'alpha': 0.1,
            'gamma': 0.9,
            'epsilon': 0.1
        },
        mcs_params={'simulations': 100}
    )
    for key in tqdm(access_pattern, desc="Memory Usage Measurement"):
        tree.access(key)
    mem_after = process.memory_info().rss / (1024 ** 2)  # in MB
    memory_used = mem_after - mem_before
    logger.info(f"Memory Usage: {memory_used:.4f} MB.")
    return memory_used

def statistical_significance_tests(ablation_results: dict, base_metrics: dict) -> dict:
    """
    Performs statistical tests to determine significance of performance improvements.

    Parameters:
        ablation_results (dict): Results from ablation studies.
        base_metrics (dict): Baseline performance metrics.

    Returns:
        dict: Statistical test results.
    """
    logger.info("Performing statistical significance tests.")
    stats_results = {}
    base_avg_depth = base_metrics.get('avg_depth', 0)
    base_total_rotations = base_metrics.get('total_rotations', 0)
    base_cache_hit_rate = base_metrics.get('cache_hit_rate', 0)

    for ablation, metrics in ablation_results.items():
        # Example: Compare avg_depth with baseline using t-test
        t_stat, p_value = stats.ttest_ind([metrics['avg_depth']], [base_avg_depth], equal_var=False)
        stats_results[ablation] = {
            't_stat': t_stat,
            'p_value': p_value
        }
        logger.debug(f"Statistical Test for {ablation}: t_stat = {t_stat:.4f}, p_value = {p_value:.4f}")
    logger.info("Statistical significance tests completed.")
    return stats_results

def interaction_effects_analysis(ablation_results: dict) -> pd.DataFrame:
    """
    Analyzes interaction effects between different hyperparameters.

    Parameters:
        ablation_results (dict): Results from ablation studies.

    Returns:
        pd.DataFrame: Correlation matrix including interaction terms.
    """
    logger.info("Analyzing interaction effects between hyperparameters.")
    data = []
    for ablation, metrics in ablation_results.items():
        entry = metrics['configuration'].copy()
        entry.update({
            'avg_depth': metrics['avg_depth'],
            'total_rotations': metrics['total_rotations'],
            'cache_hit_rate': metrics['cache_hit_rate']
        })
        data.append(entry)
    df = pd.DataFrame(data)
    
    # Example: Interaction between cache_size and alpha
    df['cache_alpha'] = df['cache_size'] * df['alpha']
    df['cache_gamma'] = df['cache_size'] * df['gamma']
    df['cache_epsilon'] = df['cache_size'] * df['epsilon']
    df['alpha_gamma'] = df['alpha'] * df['gamma']
    df['alpha_epsilon'] = df['alpha'] * df['epsilon']
    df['gamma_epsilon'] = df['gamma'] * df['epsilon']
    
    correlation_matrix = df.corr()
    correlation_matrix.to_csv('results/interaction_effects.csv')
    logger.info("Interaction effects analysis completed and saved.")
    return correlation_matrix

def robustness_analysis(tree_class, patterns: List[str], size: int, n_samples: int) -> dict:
    """
    Tests the system's robustness across different and noisy access patterns.

    Parameters:
        tree_class (class): The tree class to instantiate.
        patterns (List[str]): List of pattern types to test.
        size (int): Range of keys (0 to size-1).
        n_samples (int): Number of accesses to generate.

    Returns:
        dict: Robustness analysis results.
    """
    logger.info("Conducting robustness analysis.")
    robustness_results = {}
    for pattern in patterns:
        logger.info(f"Testing robustness with pattern: {pattern}")
        access_pattern = generate_access_pattern(pattern, size, n_samples)
        tree = tree_class(
            use_mcs=True,
            use_qlearning=True,
            use_cache=True,
            cache_size=256,
            q_params={
                'alpha': 0.1,
                'gamma': 0.9,
                'epsilon': 0.1
            },
            mcs_params={'simulations': 100}
        )
        embed = initialize_embeddings("unclemusclez/jina-embeddings-v2-base-code:latest")
        embed_and_store(pd.DataFrame({'id': range(len(access_pattern)), 'text': access_pattern}), embed, tree)
        depths = [tree._get_depth(node) for node in traverse_tree(tree.root)]
        avg_depth = np.mean(depths) if depths else 0
        total_rotations = tree.total_rotations
        cache_stats = tree.get_cache_stats() if hasattr(tree, 'get_cache_stats') else {}
        cache_hit_rate = cache_stats.get('hit_rate', 0)
        robustness_results[pattern] = {
            'avg_depth': avg_depth,
            'total_rotations': total_rotations,
            'cache_hit_rate': cache_hit_rate
        }
        logger.info(f"Robustness Results for {pattern}: Avg Depth = {avg_depth:.4f}, "
                    f"Total Rotations = {total_rotations}, Cache Hit Rate = {cache_hit_rate:.4f}")
        # Clean up
        del tree
        gc.collect()
    logger.info("Robustness analysis completed.")
    return robustness_results

# Helper function to generate access patterns
def generate_access_pattern(pattern_type: str, size: int, n: int) -> List[str]:
    """
    Generates different types of access patterns for experimentation.

    Parameters:
        pattern_type (str): Type of access pattern to generate.
        size (int): Range of keys (0 to size-1).
        n (int): Number of accesses to generate.

    Returns:
        List[str]: List of access keys as strings.
    """
    logger.debug(f"Generating access pattern: {pattern_type}, Size: {size}, Number of accesses: {n}")
    if pattern_type == 'uniform':
        pattern = np.random.choice([str(i) for i in range(size)], n)
    elif pattern_type == 'skewed':
        try:
            probabilities = np.random.zipf(2, size)
            probabilities = probabilities / probabilities.sum()
            pattern = np.random.choice([str(i) for i in range(size)], n, p=probabilities)
        except Exception as e:
            logger.error(f"Error generating skewed pattern: {e}")
            pattern = np.random.choice([str(i) for i in range(size)], n)
    elif pattern_type == 'temporal':
        access_pattern = []
        recent_items = []
        for _ in range(n):
            if recent_items and np.random.rand() < 0.7:
                access_pattern.append(np.random.choice(recent_items))
            else:
                key = str(np.random.randint(0, size))
                access_pattern.append(key)
                recent_items.append(key)
                if len(recent_items) > 100:
                    recent_items.pop(0)
        pattern = np.array(access_pattern)
    elif pattern_type == 'zipfian':
        try:
            probabilities = np.random.zipf(2, size)
            probabilities = probabilities / probabilities.sum()
            pattern = np.random.choice([str(i) for i in range(size)], n, p=probabilities)
        except Exception as e:
            logger.error(f"Error generating zipfian pattern: {e}")
            pattern = np.random.choice([str(i) for i in range(size)], n)
    elif pattern_type == 'cluster-based':
        cluster_center = np.random.choice(range(size))
        cluster_range = max(0, cluster_center - 10), min(size, cluster_center + 10)
        pattern = np.random.choice([str(i) for i in range(cluster_range[0], cluster_range[1])], n)
    elif pattern_type == 'random_walk':
        access_pattern = [str(np.random.randint(0, size))]
        for _ in range(n - 1):
            next_node = int(access_pattern[-1]) + np.random.choice([-1, 1])
            next_node = max(0, min(size - 1, next_node))
            access_pattern.append(str(next_node))
        pattern = np.array(access_pattern)
    elif pattern_type == 'bursty':
        burst_prob = 0.8
        access_pattern = []
        last_accessed = None
        for _ in range(n):
            if last_accessed is not None and np.random.rand() < burst_prob:
                access_pattern.append(last_accessed)
            else:
                last_accessed = str(np.random.randint(0, size))
                access_pattern.append(last_accessed)
        pattern = np.array(access_pattern)
    else:
        logger.warning(f"Unknown pattern type: {pattern_type}. Defaulting to uniform pattern.")
        pattern = np.random.choice([str(i) for i in range(size)], n)
    logger.debug(f"Access pattern generated with {len(pattern)} accesses.")
    return pattern.tolist()

# ==========================
# 7. Comprehensive Analysis Functions
# ==========================

def learning_curve_visualization(tree_class, access_pattern: List[str]):
    """
    Visualizes the learning curve of the Q-Learning agent over time.

    Parameters:
        tree_class (class): The tree class to instantiate.
        access_pattern (List[str]): The sequence of keys to access.

    Returns:
        None
    """
    logger.info("Generating learning curve visualization.")
    tree = tree_class(
        use_mcs=True,
        use_qlearning=True,
        use_cache=True,
        cache_size=256,
        q_params={
            'alpha': 0.1,
            'gamma': 0.9,
            'epsilon': 0.1
        },
        mcs_params={'simulations': 100}
    )
    rewards = []
    cumulative_rewards = []
    cumulative = 0
    for key in tqdm(access_pattern, desc="Learning Curve Visualization"):
        state = tree._get_state(tree._find_node(key))
        action = tree.q_agent.choose_action(state) if tree.q_agent else 'splay'
        # Simulate reward based on action
        if action == 'splay':
            reward = -tree._get_depth(tree.access(key))
        elif action == 'cache':
            reward = 1
        else:
            reward = 0
        rewards.append(reward)
        cumulative += reward
        cumulative_rewards.append(cumulative)
        # Update Q-Learning agent
        if tree.q_agent:
            next_state = tree._get_state(tree._find_node(key))
            tree.q_agent.learn(state, action, reward, next_state)

    # Plot learning curve
    plt.figure(figsize=(10,6))
    plt.plot(cumulative_rewards, label='Cumulative Reward')
    plt.title('Q-Learning Agent Learning Curve')
    plt.xlabel('Number of Accesses')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/visualizations/learning_curve.png')
    plt.close()
    logger.info("Learning curve visualization saved as 'learning_curve.png'.")

def profiling_analysis(tree_class, access_pattern: List[str]):
    """
    Profiles the tree operations to identify performance bottlenecks.

    Parameters:
        tree_class (class): The tree class to instantiate.
        access_pattern (List[str]): The sequence of keys to access.

    Returns:
        None
    """
    logger.info("Starting profiling analysis.")
    profiler = cProfile.Profile()
    profiler.enable()
    
    tree = tree_class(
        use_mcs=True,
        use_qlearning=True,
        use_cache=True,
        cache_size=256,
        q_params={
            'alpha': 0.1,
            'gamma': 0.9,
            'epsilon': 0.1
        },
        mcs_params={'simulations': 100}
    )
    embed = initialize_embeddings("unclemusclez/jina-embeddings-v2-base-code:latest")
    embed_and_store(pd.DataFrame({'id': range(len(access_pattern)), 'text': access_pattern}), embed, tree)
    for key in access_pattern[:1000]:  # Profile on a subset
        tree.access(key)
    
    profiler.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(10)  # Print top 10 functions
    
    with open('results/profiling_report.txt', 'w') as f:
        f.write(s.getvalue())
    
    logger.info("Profiling analysis completed and report saved as 'profiling_report.txt'.")

def cache_behavior_visualization(tree_class, access_pattern: List[str]):
    """
    Visualizes cache occupancy and eviction patterns over time.

    Parameters:
        tree_class (class): The tree class to instantiate.
        access_pattern (List[str]): The sequence of keys to access.

    Returns:
        None
    """
    logger.info("Generating cache behavior visualization.")
    tree = tree_class(
        use_mcs=True,
        use_qlearning=True,
        use_cache=True,
        cache_size=100,
        q_params={
            'alpha': 0.1,
            'gamma': 0.9,
            'epsilon': 0.1
        },
        mcs_params={'simulations': 100}
    )
    cache_sizes_over_time = []
    for key in tqdm(access_pattern[:1000], desc="Cache Behavior Visualization"):
        tree.access(key)
        if tree.use_cache:
            current_cache_size = len(tree.cached_nodes)
            cache_sizes_over_time.append(current_cache_size)
    
    # Plot cache occupancy over time
    plt.figure(figsize=(10,6))
    plt.plot(cache_sizes_over_time, label='Cache Size Over Time')
    plt.title('Cache Occupancy Over Time')
    plt.xlabel('Number of Accesses')
    plt.ylabel('Cache Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/visualizations/cache_behavior.png')
    plt.close()
    logger.info("Cache behavior visualization saved as 'cache_behavior.png'.")

def stability_over_multiple_runs(tree_class, access_pattern: List[str], n_runs: int =5):
    """
    Ensures consistent performance across multiple experiment runs.

    Parameters:
        tree_class (class): The tree class to instantiate.
        access_pattern (List[str]): The sequence of keys to access.
        n_runs (int): Number of runs to perform.

    Returns:
        None
    """
    logger.info(f"Assessing stability over {n_runs} runs.")
    performance_scores = []
    for run in range(1, n_runs +1):
        logger.info(f"Starting run {run}/{n_runs}.")
        tree = tree_class(
            use_mcs=True,
            use_qlearning=True,
            use_cache=True,
            cache_size=256,
            q_params={
                'alpha': 0.1,
                'gamma': 0.9,
                'epsilon': 0.1
            },
            mcs_params={'simulations': 100}
        )
        embed = initialize_embeddings("unclemusclez/jina-embeddings-v2-base-code:latest")
        embed_and_store(pd.DataFrame({'id': range(len(access_pattern)), 'text': access_pattern}), embed, tree)
        depths = [tree._get_depth(node) for node in traverse_tree(tree.root)]
        avg_depth = np.mean(depths) if depths else 0
        total_rotations = tree.total_rotations
        cache_stats = tree.get_cache_stats() if hasattr(tree, 'get_cache_stats') else {}
        cache_hit_rate = cache_stats.get('hit_rate', 0)
        performance_score = (
            0.4 * avg_depth +
            0.3 * (total_rotations / len(access_pattern)) +
            0.3 * (1 - cache_hit_rate)
        )
        performance_scores.append(performance_score)
        logger.info(f"Run {run}: Performance Score = {performance_score:.4f}")
        # Clean up
        del tree
        gc.collect()
    
    mean_score = np.mean(performance_scores)
    std_score = np.std(performance_scores)
    with open('results/stability_report.txt', 'w') as f:
        f.write(f"Stability Over {n_runs} Runs:\n")
        f.write(f"Mean Performance Score: {mean_score}\n")
        f.write(f"Standard Deviation: {std_score}\n")
    logger.info(f"Stability assessment completed. Mean Score: {mean_score:.4f}, Std Dev: {std_score:.4f}")

# ==========================
# 8. Main Execution Flow
# ==========================

def main():
    """
    Main function to run all experiments and generate visualizations and logs.
    """
    logger.info("=== Starting Comprehensive Experiments ===")
    
    # Create necessary directories
    os.makedirs('results/visualizations', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    # Paths
    tree_save_path = 'results/embeddings.pkl'  # Path to save the splay tree
    
    # Initialize embeddings
    model_name = "unclemusclez/jina-embeddings-v2-base-code:latest"
    embed = initialize_embeddings(model_name)
    
    # Load dataset from Hugging Face with selected diverse configs
    dataset_name = 'THUDM/LongBench'
    selected_configs = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", 
            "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    split = 'test'  # Specify the correct split
    df = load_dataset_from_hf(dataset_name, selected_configs)
    
    # Initialize or load splay tree
    if os.path.exists(tree_save_path):
        logger.info(f"Loading existing splay tree from '{tree_save_path}'.")
        tree = load_splay_tree(tree_save_path)
    else:
        logger.info("Initializing new splay tree.")
        base_config = {
            'use_mcs': True,
            'use_qlearning': True,
            'use_cache': True,
            'cache_size': 256,
            'alpha': 0.1,
            'gamma': 0.9,
            'epsilon': 0.1,
            'mcs_sims': 100
        }
        tree = initialize_splay_tree(base_config)
        
        # Embed and store data
        embed_and_store(df, embed, tree)
        
        # Save the tree
        save_splay_tree(tree, tree_save_path)
    
    # Perform a semantic search example
    query = "What is the significance of artificial intelligence in modern technology?"
    top_similar = semantic_search(tree, embed, query, top_n=5)
    
    # Display results
    logger.info(f"Top {len(top_similar)} similar texts to the query:")
    for text_id, score in top_similar:
        logger.info(f"Text ID: {text_id}, Similarity Score: {score:.4f}")
    
    # ==========================
    # Experimentation and Ablation Studies
    # ==========================
    
    # Define base configuration
    base_config = {
        'use_mcs': True,
        'use_qlearning': True,
        'use_cache': True,
        'cache_size': 256,
        'alpha': 0.1,
        'gamma': 0.9,
        'epsilon': 0.1,
        'mcs_sims': 100
    }
    
    # Define ablation configurations
    ablation_configs = [
        {'use_mcs': False},  # Disable Monte Carlo Search
        {'use_qlearning': False},  # Disable Q-Learning
        {'use_cache': False},  # Disable Caching
        {'use_mcs': False, 'use_qlearning': False},  # Disable both MCS and Q-Learning
        {'use_qlearning': False, 'use_cache': False},  # Disable Q-Learning and Caching
        {'use_mcs': False, 'use_cache': False},  # Disable MCS and Caching
        {'use_mcs': False, 'use_qlearning': False, 'use_cache': False}  # Disable all
    ]
    
    # Perform ablation studies
    ablation_results = perform_ablation_study(df, embed, base_config, ablation_configs, tree_save_path)
    
    # Save ablation results
    save_results(ablation_results, 'results/ablation_studies.json')
    
    # ==========================
    # Comprehensive Analyses
    # ==========================
    
    # Parameter Sensitivity Analysis
    sensitivity_stats = parameter_sensitivity_analysis(ablation_results)
    save_results(sensitivity_stats, 'results/parameter_sensitivity.json')
    
    # Correlation Analysis
    pearson_corr, spearman_corr = correlation_analysis(ablation_results)
    
    # Runtime Performance Measurement
    runtime = runtime_performance_measurement(MCSTSQSplayTree, df['id'].astype(str).tolist())
    runtime_results = {'runtime_seconds': runtime}
    save_results(runtime_results, 'results/runtime_performance.json')
    
    # Memory Usage Analysis
    memory_used = memory_usage_analysis(MCSTSQSplayTree, df['id'].astype(str).tolist())
    memory_results = {'memory_used_mb': memory_used}
    save_results(memory_results, 'results/memory_usage.json')
    
    # Statistical Significance Tests
    # Assuming base_metrics as the initial tree metrics
    depths = [tree._get_depth(node) for node in traverse_tree(tree.root)]
    avg_depth = np.mean(depths) if depths else 0
    total_rotations = tree.total_rotations
    cache_stats = tree.get_cache_stats() if hasattr(tree, 'get_cache_stats') else {}
    cache_hit_rate = cache_stats.get('hit_rate', 0)
    base_metrics = {
        'avg_depth': avg_depth,
        'total_rotations': total_rotations,
        'cache_hit_rate': cache_hit_rate
    }
    stats_results = statistical_significance_tests(ablation_results, base_metrics)
    save_results(stats_results, 'results/statistical_tests.json')
    
    # Interaction Effects Analysis
    interaction_corr = interaction_effects_analysis(ablation_results)
    
    # Robustness Analysis
    robustness_patterns = ['uniform', 'skewed', 'random_walk']
    robustness_results = robustness_analysis(MCSTSQSplayTree, robustness_patterns, 1000, 5000)
    save_results(robustness_results, 'results/robustness_analysis.json')
    
    # Learning Curve Visualization
    learning_curve_visualization(MCSTSQSplayTree, df['id'].astype(str).tolist())
    
    # Profiling Analysis
    profiling_analysis(MCSTSQSplayTree, df['id'].astype(str).tolist())
    
    # Cache Behavior Visualization
    cache_behavior_visualization(MCSTSQSplayTree, df['id'].astype(str).tolist())
    
    # Stability Over Multiple Runs
    stability_over_multiple_runs(MCSTSQSplayTree, df['id'].astype(str).tolist(), n_runs=5)
    
    logger.info("=== All experiments and analyses completed successfully! ===")

if __name__ == "__main__":
    main()
