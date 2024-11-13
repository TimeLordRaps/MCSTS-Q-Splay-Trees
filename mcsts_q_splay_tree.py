# mcsts_q_splay_tree.py
import logging
import numpy as np
from splay_tree import SplayTree, Node
import random
from collections import OrderedDict

class QLearningAgent:
    """
    Q-Learning agent that learns splay and caching policies over time.
    """
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Maps states to action values
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.actions = actions  # Actions include ['splay', 'cache', 'none']

    def choose_action(self, state):
        """Selects an action based on the current state using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        q_values = self.q_table.get(state, {})
        if not q_values:
            return np.random.choice(self.actions)
        return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state):
        """Updates the Q-value for the given state-action pair."""
        q_values = self.q_table.setdefault(state, {a: 0.0 for a in self.actions})
        max_future_q = max(self.q_table.get(next_state, {}).values(), default=0.0)
        q_values[action] += self.alpha * (reward + self.gamma * max_future_q - q_values[action])

class MonteCarloSearch:
    """
    Monte Carlo Search class to evaluate possible actions for selective splay operations.
    """
    def __init__(self, simulations=100):
        self.simulations = simulations

    def optimize_splay(self, tree, node):
        """Uses Monte Carlo simulations to optimize splay decisions."""
        best_action = None
        lowest_cost = float('inf')
        for action in ['no_splay', 'partial_splay', 'full_splay']:
            total_cost = 0
            for _ in range(self.simulations):
                cost = self.simulate_action(tree, node, action)
                total_cost += cost
            average_cost = total_cost / self.simulations
            if average_cost < lowest_cost:
                lowest_cost = average_cost
                best_action = action
        if best_action == 'partial_splay':
            tree.partial_splay(node)
        elif best_action == 'full_splay':
            tree._splay(node)

    def simulate_action(self, tree, node, action):
        """Simulates the cost of performing a specific splay action."""
        if action == 'no_splay':
            return tree._get_depth(node)
        elif action == 'partial_splay':
            return tree._get_depth(node) / 2
        elif action == 'full_splay':
            return 0
        return float('inf')

class MCSTSQSplayTree(SplayTree):
    """
    Enhanced splay tree with Monte Carlo Search, Q-Learning, and caching mechanism.
    """
    def __init__(self, use_mcs=True, use_qlearning=True, use_cache=True, cache_size=100, q_params=None, mcs_params=None):
        super().__init__()
        self.use_mcs = use_mcs
        self.use_qlearning = use_qlearning
        self.use_cache = use_cache
        self.cache_size = cache_size
        # Use OrderedDict instead of set for LRU cache functionality
        self.cached_nodes = OrderedDict() if use_cache else None
        self.q_agent = QLearningAgent(actions=['splay', 'cache', 'none'], **(q_params or {})) if use_qlearning else None
        self.mcs = MonteCarloSearch(**(mcs_params or {})) if use_mcs else None
        # Initialize metrics tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_accesses = 0

    def access(self, key):
        """Accesses a node, deciding whether to splay, cache, or take no action based on learned strategies."""
        self.total_accesses += 1

        if self.use_cache and key in self.cached_nodes:
            # Cache hit - move to end to mark as most recently used
            self.cached_nodes.move_to_end(key)
            self.cache_hits += 1
            return self.search(key)  # Return the value associated with the key

        self.cache_misses += 1
        node = self._find_node(key)
        if node is None:
            return None

        if self.use_qlearning and self.q_agent:
            state = self._get_state(node)
            action = self.q_agent.choose_action(state)
        else:
            action = 'splay'

        reward = 0
        if action == 'splay':
            if self.use_mcs and self.mcs:
                self.mcs.optimize_splay(self, node)
            else:
                self._splay(node)
            reward = -self._get_depth(node)
        elif action == 'cache' and self.use_cache:
            self._cache_node(key)
            reward = 1

        if self.use_qlearning and self.q_agent:
            next_state = self._get_state(node)
            self.q_agent.learn(state, action, reward, next_state)

        return node.value  # Return the value after action

    def _cache_node(self, key):
        """Adds a node to the cache, evicting least recently used if necessary."""
        if self.use_cache:
            if len(self.cached_nodes) >= self.cache_size:
                # Remove least recently used item
                evicted_key, _ = self.cached_nodes.popitem(last=False)
                # logger.debug(f"Evicted key from cache: {evicted_key}")
            self.cached_nodes[key] = True
            # logger.debug(f"Cached key: {key}")

    def _get_state(self, node):
        """Generates the state representation for the Q-Learning agent."""
        if node:
            depth = self._get_depth(node)
            is_cached = 1 if node.key in self.cached_nodes else 0
            return (depth, is_cached)
        return ('not_found', 0)

    def get_cache_stats(self):
        """Returns cache performance statistics."""
        if not self.use_cache or self.total_accesses == 0:
            return {"hit_rate": 0, "miss_rate": 0}
        hit_rate = self.cache_hits / self.total_accesses
        miss_rate = self.cache_misses / self.total_accesses
        return {
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "total_accesses": self.total_accesses,
            "cache_size": len(self.cached_nodes),
            "max_cache_size": self.cache_size
        }

    def partial_splay(self, node):
        """Performs a partial splay to bring the node halfway towards the root."""
        current = node
        for _ in range(self._get_depth(node) // 2):
            if current.parent is None:
                break
            if current == current.parent.left:
                self._right_rotate(current.parent)
            else:
                self._left_rotate(current.parent)
            current = current.parent
