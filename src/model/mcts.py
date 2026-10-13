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
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[Move, MCTSNode] = {}
        self.is_expanded = False

    def Q(self) -> float:
        """Mean value from this node's side-to-move."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def U(self, c_puct: float, parent_visit_count: int) -> float:
        """Upper confidence bound. Parent N is at least 1 so priors matter on the first visit."""
        parent_visits = max(parent_visit_count, 1)
        return c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)

    def best_child(self, c_puct: float) -> 'MCTSNode':
        """Select the child with the highest PUCT score for this node's player.

        Q is stored from the child's side-to-move (the opponent), so the parent
        maximises `-child.Q() + U`.
        """
        return max(
            self.children.values(),
            key=lambda child: -child.Q() + child.U(c_puct, self.visit_count),
        )

    def select_action(self, temperature: float = 1.0) -> Optional[Move]:
        """Select a move from visit counts. temperature=0 is greedy."""
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not self.children:
            return None

        visit_counts = np.array(
            [child.visit_count for child in self.children.values()],
            dtype=np.float64,
        )
        moves = list(self.children.keys())
        if temperature == 0 or visit_counts.sum() <= 0:
            return moves[int(np.argmax(visit_counts))]

        log_weights = np.full_like(visit_counts, -np.inf)
        positive = visit_counts > 0
        log_weights[positive] = np.log(visit_counts[positive])
        log_weights = (log_weights - log_weights.max()) / temperature
        scaled = np.exp(log_weights)
        total = scaled.sum()
        if total <= 0 or not np.isfinite(total):
            return moves[int(np.argmax(visit_counts))]
        probs = scaled / total
        return moves[int(np.random.choice(len(moves), p=probs))]

    def get_policy_target(self) -> np.ndarray:
        """Normalized visit counts as an 1800-d training target."""
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if not self.children:
            return policy

        visit_counts = np.array(
            [child.visit_count for child in self.children.values()],
            dtype=np.float32,
        )
        total_visits = visit_counts.sum()
        if total_visits <= 0:
            return policy

        visit_counts = visit_counts / total_visits
        for move, probability in zip(self.children, visit_counts):
            policy[Rules.move_to_action_index(*move, board=self.game_state.board)] = probability
        return policy


class MCTS:
    """Monte Carlo Tree Search"""

    def __init__(
        self,
        model: ChineseChessNet,
        num_simulations: int = 400,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        if type(num_simulations) is not int or num_simulations <= 0:
            raise ValueError("num_simulations must be a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(c_puct) or c_puct <= 0:
            raise ValueError("c_puct must be finite and positive")
        if not math.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be finite and positive")
        if not 0 <= dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be between 0 and 1")

        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, game_state: GameState) -> Tuple[Optional[Move], np.ndarray]:
        """Run MCTS and return (best_move, policy_target[1800])."""
        if game_state.is_terminal():
            return None, np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        was_training = self.model.training
        self.model.eval()
        try:
            root = MCTSNode(game_state)
            self._expand_node(root, add_noise=self.dirichlet_epsilon > 0)
            for _ in range(self.num_simulations):
                self._simulate(root)
            return root.select_action(temperature=self.temperature), root.get_policy_target()
        finally:
            self.model.train(was_training)

    def _simulate(self, node: MCTSNode):
        path = self._select_path(node)
        value = self._expand_and_evaluate(path[-1])
        self._backup_value(path, value)

    def _select_path(self, root: MCTSNode) -> List[MCTSNode]:
        path = [root]
        current = root
        while current.is_expanded and not current.game_state.is_terminal():
            if not current.children:
                break
            current = current.best_child(self.c_puct)
            path.append(current)
        return path

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        if node.game_state.is_terminal():
            return node.game_state.get_game_result()

        policy, value = self._infer(node)
        if not node.is_expanded:
            self._expand_node(node, add_noise=False, policy=policy)
        return value

    def _infer(self, node: MCTSNode) -> Tuple[np.ndarray, float]:
        """One network forward: legal-move policy over 1800 and side-to-move value."""
        board_tensor = torch.from_numpy(node.game_state.to_tensor()).unsqueeze(0)
        board_tensor = board_tensor.to(self.model.get_device())
        with torch.no_grad():
            policy_logits, value_tensor = self.model(board_tensor)
            logits = policy_logits.float().cpu().numpy()[0]
            value = float(value_tensor.item())
            if logits.shape != (ACTION_SPACE_SIZE,) or not np.isfinite(logits).all():
                raise ValueError("Network policy logits must be finite with shape (1800,)")
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError("Network value must be finite and within [-1, 1]")
            return logits, value

    def _backup_value(self, path: List[MCTSNode], value: float):
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value
            value = -value

    def _expand_node(
        self,
        node: MCTSNode,
        add_noise: bool = False,
        policy: Optional[np.ndarray] = None,
    ):
        if node.is_expanded:
            return

        legal_moves = node.game_state.get_legal_moves()
        if not legal_moves:
            node.is_expanded = True
            return

        if policy is None:
            policy, _ = self._infer(node)
        move_priors = self._priors_from_policy(policy, legal_moves, node.game_state)
        if add_noise:
            move_priors = self._add_dirichlet_noise(move_priors)
        self._create_child_nodes(node, move_priors)
        node.is_expanded = True

    def _priors_from_policy(
        self,
        policy: np.ndarray,
        legal_moves: List[Move],
        game_state: GameState,
    ) -> List[Tuple[Move, float]]:
        """Apply a stable softmax only to legal-action logits."""
        indices = [Rules.move_to_action_index(*move, board=game_state.board)
                   for move in legal_moves]
        if len(indices) != len(set(indices)):
            raise RuntimeError("Legal moves collided in the action encoder")
        legal_logits = np.asarray(policy[indices], dtype=np.float64)
        if not np.isfinite(legal_logits).all():
            raise ValueError("Legal policy logits must be finite")
        weights = np.exp(legal_logits - legal_logits.max())
        probabilities = weights / weights.sum()
        return list(zip(legal_moves, probabilities.tolist()))

    def _normalize_priors(
        self,
        move_priors: List[Tuple[Move, float]],
    ) -> List[Tuple[Move, float]]:
        total_prior = sum(prior for _, prior in move_priors)
        if np.isfinite(total_prior) and total_prior > 0:
            return [(move, prior / total_prior) for move, prior in move_priors]
        uniform_prior = 1.0 / len(move_priors)
        return [(move, uniform_prior) for move, _ in move_priors]

    def _add_dirichlet_noise(
        self,
        move_priors: List[Tuple[Move, float]],
    ) -> List[Tuple[Move, float]]:
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(move_priors))
        return [
            (move, (1 - self.dirichlet_epsilon) * prior + self.dirichlet_epsilon * n)
            for (move, prior), n in zip(move_priors, noise)
        ]

    def _create_child_nodes(
        self,
        parent: MCTSNode,
        move_priors: List[Tuple[Move, float]],
    ):
        for move, prior in move_priors:
            child_state = parent.game_state.copy()
            if not child_state.make_move(*move):
                raise RuntimeError(f"Legal move failed while expanding MCTS: {move}")
            parent.children[move] = MCTSNode(
                child_state, parent=parent, move=move, prior=prior
            )

    def get_action_probs(
        self,
        game_state: GameState,
        temperature: float = 1.0,
    ) -> Tuple[Optional[Move], np.ndarray]:
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        old_temp = self.temperature
        self.temperature = temperature
        try:
            return self.search(game_state)
        finally:
            self.temperature = old_temp
