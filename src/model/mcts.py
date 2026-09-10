"""
Monte Carlo Tree Search (MCTS) cho Chinese Chess
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..game.game_state import GameState
from ..game.rules import ACTION_SPACE_SIZE, Move, Rules
from .neural_network import ChineseChessNet

class MCTSNode:
    """Node trong cây MCTS"""
    
    def __init__(
        self,
        game_state: GameState,
        parent: Optional['MCTSNode'] = None,
        move: Optional[Move] = None,
        prior: float = 0.0,
    ):
        """
        Args:
            game_state: Trạng thái game tại node này
            parent: Node cha
            move: Nước đi từ parent đến node này
            prior: Xác suất prior từ policy network
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move  # (from_row, from_col, to_row, to_col)
        self.prior = prior
        
        # Statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[Move, MCTSNode] = {}
        self.is_expanded = False
    
    def Q(self) -> float:
        """Mean action value"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def U(self, c_puct: float, parent_visit_count: int) -> float:
        """Upper confidence bound"""
        return c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
    
    def best_child(self, c_puct: float) -> 'MCTSNode':
        """Chọn con tốt nhất theo UCB"""
        return max(self.children.values(), 
                  key=lambda child: child.Q() + child.U(c_puct, self.visit_count))
    
    def select_action(self, temperature: float = 1.0) -> Move:
        """
        Chọn action sau khi MCTS hoàn thành
        
        Args:
            temperature: 0 = greedy, 1 = stochastic
        """
        if not self.children:
            return None
        
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        moves = list(self.children.keys())
        
        if temperature == 0:
            # Greedy - chọn nước đi có visit count cao nhất
            action_idx = np.argmax(visit_counts)
        else:
            # Stochastic - sample theo visit count
            visit_counts = visit_counts ** (1.0 / temperature)
            probs = visit_counts / visit_counts.sum()
            action_idx = np.random.choice(len(moves), p=probs)
        
        return moves[action_idx]
    
    def get_policy_target(self) -> np.ndarray:
        """
        Lấy target policy (normalized visit counts) để train network
        
        Returns:
            numpy array shape (1800,) - only legal moves have non-zero probs
        """
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        
        if not self.children:
            return policy
        
        visit_counts = []
        for move, child in self.children.items():
            visit_counts.append(child.visit_count)
        
        # Normalize visit counts to probabilities
        visit_counts = np.array(visit_counts, dtype=np.float32)
        total_visits = visit_counts.sum()
        if total_visits <= 0:
            return policy

        visit_counts = visit_counts / total_visits
        
        # Multiple legal moves can share an action bucket in the legacy 1800
        # action space. Accumulation keeps the target normalized.
        for move, probability in zip(self.children, visit_counts):
            action_idx = Rules.move_to_action_index(*move)
            policy[action_idx] += probability
        
        return policy

class MCTS:
    """Monte Carlo Tree Search"""
    
    def __init__(self, model: ChineseChessNet, num_simulations: int = 400, 
                 c_puct: float = 1.5, temperature: float = 1.0,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25):
        """
        Args:
            model: Neural network
            num_simulations: Số lần simulation cho mỗi lần search
            c_puct: Exploration constant
            temperature: Temperature cho action selection
            dirichlet_alpha: Alpha cho Dirichlet noise
            dirichlet_epsilon: Epsilon cho Dirichlet noise
        """
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if c_puct <= 0:
            raise ValueError("c_puct must be positive")
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0 <= dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be between 0 and 1")

        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
    
    def search(self, game_state: GameState) -> Tuple[Optional[Move], np.ndarray]:
        """
        Thực hiện MCTS search
        
        Args:
            game_state: Trạng thái game hiện tại
        
        Returns:
            best_move: (from_row, from_col, to_row, to_col)
            policy_target: numpy array shape (1800,)
        """
        was_training = self.model.training
        self.model.eval()
        try:
            root = MCTSNode(game_state)

            # Add Dirichlet noise to root for exploration.
            self._expand_node(root, add_noise=True)

            for _ in range(self.num_simulations):
                self._simulate(root)

            best_move = root.select_action(temperature=self.temperature)
            policy_target = root.get_policy_target()
            return best_move, policy_target
        finally:
            self.model.train(was_training)
    
    def _simulate(self, node: MCTSNode):
        """Run one MCTS simulation."""
        # 1. Selection - traverse tree using UCB
        path = self._select_path(node)
        current = path[-1]
        
        # 2. Expansion & Evaluation
        value = self._expand_and_evaluate(current)
        
        # 3. Backup - propagate value up the tree
        self._backup_value(path, value)
    
    def _select_path(self, root: MCTSNode) -> List[MCTSNode]:
        """Select path from root to leaf using UCB."""
        path = [root]
        current = root
        
        while current.is_expanded and not current.game_state.is_terminal():
            if not current.children:
                break
            current = current.best_child(self.c_puct)
            path.append(current)
        
        return path
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Expand node if needed and get value from network."""
        if node.game_state.is_terminal():
            return node.game_state.get_game_result()
        
        if not node.is_expanded:
            self._expand_node(node, add_noise=False)
        
        return self._evaluate_position(node)
    
    def _evaluate_position(self, node: MCTSNode) -> float:
        """Get position evaluation from neural network."""
        board_tensor = torch.from_numpy(node.game_state.to_tensor()).unsqueeze(0)
        board_tensor = board_tensor.to(self.model.get_device())
        
        with torch.no_grad():
            _, value_tensor = self.model(board_tensor)
            return value_tensor.item()
    
    def _backup_value(self, path: List[MCTSNode], value: float):
        """Backup value through the path to root."""
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value
            value = -value  # Flip perspective for alternating players
    
    def _expand_node(self, node: MCTSNode, add_noise: bool = False):
        """Expand node - thêm tất cả children"""
        if node.is_expanded:
            return
        
        legal_moves = node.game_state.get_legal_moves()
        if not legal_moves:
            node.is_expanded = True
            return
        
        # Get move priors from network
        move_priors = self._get_move_priors(node, legal_moves)
        
        # Add exploration noise to root node
        if add_noise:
            move_priors = self._add_dirichlet_noise(move_priors)
        
        # Create child nodes
        self._create_child_nodes(node, move_priors)
        node.is_expanded = True
    
    def _get_move_priors(self, node: MCTSNode, 
                        legal_moves: List[Move]) -> List[Tuple[Move, float]]:
        """Get prior probabilities for legal moves from neural network."""
        board_tensor = torch.from_numpy(node.game_state.to_tensor()).unsqueeze(0)
        board_tensor = board_tensor.to(self.model.get_device())
        
        with torch.no_grad():
            policy_logits, _ = self.model(board_tensor)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        # Map legal moves to priors
        move_priors = []
        for move in legal_moves:
            action_idx = Rules.move_to_action_index(*move)
            prior = policy[action_idx]
            move_priors.append((move, prior))
        
        # Normalize priors
        return self._normalize_priors(move_priors)
    
    def _normalize_priors(self, move_priors: List[Tuple[Move, float]]) -> List[Tuple[Move, float]]:
        """Normalize prior probabilities to sum to 1."""
        total_prior = sum(prior for _, prior in move_priors)
        
        if np.isfinite(total_prior) and total_prior > 0:
            return [(move, prior / total_prior) for move, prior in move_priors]

        # Fall back to a legal uniform prior if model output is unusable.
        uniform_prior = 1.0 / len(move_priors)
        return [(move, uniform_prior) for move, _ in move_priors]
    
    def _add_dirichlet_noise(self, move_priors: List[Tuple[Move, float]]) -> List[Tuple[Move, float]]:
        """Add Dirichlet noise to move priors for exploration."""
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(move_priors))
        return [
            (move, (1 - self.dirichlet_epsilon) * prior + self.dirichlet_epsilon * n)
            for (move, prior), n in zip(move_priors, noise)
        ]
    
    def _create_child_nodes(self, parent: MCTSNode, 
                           move_priors: List[Tuple[Move, float]]):
        """Create child nodes for all legal moves."""
        for move, prior in move_priors:
            child_state = parent.game_state.copy()
            from_row, from_col, to_row, to_col = move
            child_state.make_move(from_row, from_col, to_row, to_col)
            
            child_node = MCTSNode(child_state, parent=parent, move=move, prior=prior)
            parent.children[move] = child_node
    
    def get_action_probs(self, game_state: GameState, temperature: float = 1.0) -> Tuple[Move, np.ndarray]:
        """
        Public method để lấy action và policy
        
        Returns:
            action: (from_row, from_col, to_row, to_col)
            policy: numpy array shape (1800,)
        """
        # Temporarily set temperature
        old_temp = self.temperature
        self.temperature = temperature
        try:
            return self.search(game_state)
        finally:
            self.temperature = old_temp
