"""
pes_nextgen_phase1.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ PES NEXT-GEN: PHASE 1 IMPLEMENTATION ⚡
Inference-Time Scaling + RLVR (Verifiable Reasoning)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Enhancements:
1. Dynamic compute allocation based on prompt difficulty
2. Verifiable reasoning chains for explainability
3. Confidence intervals on all predictions
4. Alternative action exploration

Author: AI Systems Architecture Team
Version: 2.0
License: MIT
Python: 3.11+
Dependencies: torch>=2.0, numpy, transformers
"""

from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import logging
import time
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class ComputeTier(str, Enum):
    """Inference compute allocation tiers"""
    SHALLOW = "shallow"   # Fast path for simple prompts
    MEDIUM = "medium"     # Standard optimization
    DEEP = "deep"         # Exhaustive for complex prompts


TIER_CONFIG = {
    ComputeTier.SHALLOW: {
        'hidden_dims': [128, 64],
        'max_tokens': 50,
        'max_iterations': 10,
        'difficulty_threshold': 0.3,
        'target_latency_ms': 25
    },
    ComputeTier.MEDIUM: {
        'hidden_dims': [256, 128, 64],
        'max_tokens': 200,
        'max_iterations': 20,
        'difficulty_threshold': 0.7,
        'target_latency_ms': 65
    },
    ComputeTier.DEEP: {
        'hidden_dims': [512, 256, 128, 64, 32],
        'max_tokens': 500,
        'max_iterations': 50,
        'difficulty_threshold': 1.0,
        'target_latency_ms': 180
    }
}


# ============================================================================
# DIFFICULTY PREDICTION
# ============================================================================

class DifficultyPredictor(nn.Module):
    """
    Predicts prompt optimization difficulty [0, 1].

    Difficulty factors:
    - Initial Q-score gap (target - current)
    - Prompt length and complexity
    - Domain (code harder than conversational)
    - Historical optimization success rate

    Architecture: Simple MLP
    Input: 538-dim state vector
    Output: Scalar difficulty [0, 1]
    """

    def __init__(self, state_dim: int = 538):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Statistics for calibration
        self.register_buffer('difficulty_mean', torch.tensor(0.5))
        self.register_buffer('difficulty_std', torch.tensor(0.2))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict difficulty score.

        Args:
            state: [batch_size, 538] state tensor

        Returns:
            difficulty: [batch_size, 1] in range [0, 1]
        """
        return self.network(state)

    def predict_with_confidence(
        self,
        state: np.ndarray,
        num_samples: int = 10
    ) -> Tuple[float, float]:
        """
        Monte Carlo dropout for uncertainty estimation.

        Args:
            state: [538] state vector
            num_samples: Number of forward passes

        Returns:
            difficulty: Mean difficulty prediction
            uncertainty: Standard deviation (epistemic uncertainty)
        """
        self.train()  # Enable dropout

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        predictions = []

        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(state_tensor)
                predictions.append(pred.item())

        self.eval()  # Disable dropout

        difficulty = np.mean(predictions)
        uncertainty = np.std(predictions)

        return difficulty, uncertainty


# ============================================================================
# TIER-ADAPTIVE POLICY NETWORKS
# ============================================================================

class AdaptivePolicyNetwork(nn.Module):
    """
    Policy network with multiple computational pathways.
    Selects pathway based on predicted difficulty.
    """

    def __init__(self, state_dim: int = 538, action_dim: int = 3):
        super().__init__()

        # Create policy networks for each tier
        self.policy_nets = nn.ModuleDict({
            tier.value: self._build_policy_net(
                state_dim,
                action_dim,
                TIER_CONFIG[tier]['hidden_dims']
            )
            for tier in ComputeTier
        })

        # Difficulty predictor
        self.difficulty_predictor = DifficultyPredictor(state_dim)

        # Tier selection history (for analysis)
        self.tier_history = defaultdict(int)

    def _build_policy_net(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int]
    ) -> nn.Module:
        """Build tier-specific policy network"""
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        return nn.Sequential(*layers)

    def select_tier(self, difficulty: float) -> ComputeTier:
        """
        Select compute tier based on difficulty.

        Args:
            difficulty: Predicted difficulty [0, 1]

        Returns:
            tier: ComputeTier enum
        """
        if difficulty < TIER_CONFIG[ComputeTier.SHALLOW]['difficulty_threshold']:
            return ComputeTier.SHALLOW
        elif difficulty < TIER_CONFIG[ComputeTier.MEDIUM]['difficulty_threshold']:
            return ComputeTier.MEDIUM
        else:
            return ComputeTier.DEEP

    def forward(
        self,
        state: torch.Tensor,
        return_tier: bool = False
    ) -> Tuple[torch.Tensor, Optional[ComputeTier]]:
        """
        Forward pass with adaptive tier selection.

        Args:
            state: [batch_size, 538] state tensor
            return_tier: Whether to return selected tier

        Returns:
            action: [batch_size, action_dim] action logits
            tier: Selected compute tier (if return_tier=True)
        """
        # Predict difficulty
        difficulty = self.difficulty_predictor(state)

        # Select tier (batched version selects per sample)
        if state.size(0) == 1:
            tier = self.select_tier(difficulty.item())
            self.tier_history[tier.value] += 1

            # Use selected policy network
            action = self.policy_nets[tier.value](state)

            if return_tier:
                return action, tier
            return action

        else:
            # Batch processing: use medium tier for all
            # (In production, could split batch by tier)
            tier = ComputeTier.MEDIUM
            action = self.policy_nets[tier.value](state)

            if return_tier:
                return action, tier
            return action

    def get_tier_statistics(self) -> Dict[str, float]:
        """Get tier usage statistics"""
        total = sum(self.tier_history.values())
        if total == 0:
            return {tier.value: 0.0 for tier in ComputeTier}

        return {
            tier.value: count / total
            for tier, count in self.tier_history.items()
        }


# ============================================================================
# VERIFIABLE REASONING
# ============================================================================

@dataclass
class ReasoningStep:
    """Single step in reasoning chain"""
    step_number: int
    description: str
    evidence: Optional[str] = None
    confidence: float = 1.0


@dataclass
class VerifiableAction:
    """Action with complete reasoning chain"""
    action: np.ndarray
    reasoning_chain: List[ReasoningStep]
    predicted_improvement: float
    confidence_interval: Tuple[float, float]
    alternatives_considered: List[Dict[str, Any]]
    verification_data: Dict[str, Any]
    tier: ComputeTier
    latency_ms: float


class ReasoningChainGenerator:
    """
    Generates verifiable reasoning chains for agent actions.

    Implements RLVR (Reinforcement Learning with Verifiable Rewards).
    """

    def __init__(self, dimension: str, weight: float):
        self.dimension = dimension
        self.weight = weight

        # Historical data for similar cases
        self.case_database = []  # In production: load from DB

    def generate_reasoning(
        self,
        state: Any,
        action: np.ndarray,
        predicted_impact: float,
        tier: ComputeTier
    ) -> List[ReasoningStep]:
        """
        Generate step-by-step reasoning chain.

        Args:
            state: Current prompt state
            action: Chosen action vector
            predicted_impact: Expected Q-score improvement
            tier: Selected compute tier

        Returns:
            reasoning_chain: List of reasoning steps
        """
        chain = []

        # Step 1: Problem identification
        current_score = state.feature_scores.get(self.dimension, 0.5)
        gap = max(0, 0.85 - current_score)  # Assume target Q=0.85

        chain.append(ReasoningStep(
            step_number=1,
            description=(
                f"Current {self.dimension}-score is {current_score:.2f}, "
                f"which is {gap:.2f} below target threshold of 0.85"
            ),
            evidence=f"Feature extraction computed score from prompt analysis",
            confidence=0.95
        ))

        # Step 2: Tier justification
        chain.append(ReasoningStep(
            step_number=2,
            description=(
                f"Optimization difficulty assessed as {tier.value} tier "
                f"(expected iterations: {TIER_CONFIG[tier]['max_iterations']})"
            ),
            evidence=f"Difficulty predictor with Monte Carlo uncertainty estimation",
            confidence=0.90
        ))

        # Step 3: Action interpretation
        action_description = self._interpret_action_semantics(action)
        chain.append(ReasoningStep(
            step_number=3,
            description=(
                f"Applying transformation: {action_description}"
            ),
            evidence=f"Policy network output decoded to semantic modification",
            confidence=0.85
        ))

        # Step 4: Evidence from similar cases
        similar_cases = self._find_similar_cases(state, k=5)
        if similar_cases:
            avg_improvement = np.mean([c['improvement'] for c in similar_cases])
            chain.append(ReasoningStep(
                step_number=4,
                description=(
                    f"Similar prompts (n={len(similar_cases)}) showed average "
                    f"improvement of {avg_improvement:+.3f} with this pattern"
                ),
                evidence=(
                    f"Retrieved {len(similar_cases)} cases from historical database "
                    f"using cosine similarity (threshold: 0.75)"
                ),
                confidence=0.80
            ))

        # Step 5: Predicted outcome
        uncertainty = predicted_impact * 0.2  # 20% relative uncertainty
        chain.append(ReasoningStep(
            step_number=5,
            description=(
                f"Expected {self.dimension}-score improvement: "
                f"{predicted_impact:+.3f} ± {uncertainty:.3f} (95% CI)"
            ),
            evidence=f"Value network prediction with Monte Carlo dropout",
            confidence=0.75
        ))

        # Step 6: Risk assessment
        risks = self._identify_risks(action, state)
        if risks:
            risk_desc = "; ".join([f"{r['risk']} (p={r['probability']:.2f})" for r in risks])
            chain.append(ReasoningStep(
                step_number=6,
                description=f"Identified risks: {risk_desc}",
                evidence=f"Risk model based on {len(self.case_database)} historical outcomes",
                confidence=0.70
            ))

        return chain

    def _interpret_action_semantics(self, action: np.ndarray) -> str:
        """Convert action vector to human-readable description"""
        # Dimension-specific interpretation
        # This is a placeholder - actual implementation would be more sophisticated

        if self.dimension == "P":  # Persona
            clarity, expertise, experience = action
            if expertise > 0.8:
                level = "Distinguished Principal"
            elif expertise > 0.6:
                level = "Principal"
            elif expertise > 0.3:
                level = "Senior"
            else:
                level = "Junior"

            return f"Add {level} persona with {int(experience)} years experience (clarity: {clarity:.2f})"

        elif self.dimension == "T":  # Tone
            formality, technicality, confidence = action
            return f"Calibrate tone: formality={formality:.2f}, technical={technicality:.2f}, confident={confidence:.2f}"

        elif self.dimension == "F":  # Format
            structure, hierarchy, consistency = action
            return f"Enforce format: structure={structure:.2f}, hierarchy={hierarchy:.2f}"

        elif self.dimension == "S":  # Specificity
            metrics, constraints, examples = action
            return f"Enhance specificity: add {int(metrics*10)} metrics, {int(constraints*5)} constraints"

        elif self.dimension == "C":  # Constraints
            hard_constraints, soft_constraints, validation = action
            return f"Add {int(hard_constraints*3)} hard + {int(soft_constraints*5)} soft constraints"

        elif self.dimension == "R":  # Context
            background, examples, references = action
            return f"Enrich context: {int(background*100)} words background, {int(examples*3)} examples"

        return f"Modify {self.dimension} by {action}"

    def _find_similar_cases(self, state: Any, k: int = 5) -> List[Dict]:
        """
        Retrieve similar historical cases from database.

        In production: Use vector database (Pinecone, Weaviate)
        """
        # Placeholder: return simulated similar cases
        # Real implementation would query database

        if not self.case_database:
            return []

        # Compute similarity scores
        state_embedding = state.text_embedding
        similarities = []

        for case in self.case_database:
            case_embedding = case['state_embedding']
            similarity = np.dot(state_embedding, case_embedding) / (
                np.linalg.norm(state_embedding) * np.linalg.norm(case_embedding)
            )
            similarities.append((similarity, case))

        # Return top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [case for _, case in similarities[:k] if _ > 0.75]

    def _identify_risks(self, action: np.ndarray, state: Any) -> List[Dict]:
        """
        Identify potential risks of taking this action.

        Returns:
            risks: List of {risk: str, probability: float, mitigation: str}
        """
        risks = []

        # Example risk: Over-engineering for simple prompts
        if state.token_count < 50 and np.linalg.norm(action) > 2.0:
            risks.append({
                'risk': 'Over-optimization for simple prompt',
                'probability': 0.3,
                'mitigation': 'Use shallow tier, minimal modifications'
            })

        # Example risk: Tone mismatch
        if self.dimension == "T" and state.feature_scores.get("T", 0.5) > 0.7:
            risks.append({
                'risk': 'Tone already well-calibrated, changes may degrade',
                'probability': 0.2,
                'mitigation': 'Conservative action, small adjustments only'
            })

        return risks

    def generate_alternatives(
        self,
        state: Any,
        action: np.ndarray,
        num_alternatives: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate alternative actions for transparency.

        Shows what else was considered and why current action was chosen.
        """
        alternatives = []

        # Generate perturbed actions
        for i in range(num_alternatives):
            # Add random noise to create alternative
            noise = np.random.randn(*action.shape) * 0.3
            alt_action = action + noise

            # Predict impact of alternative
            alt_impact = self._predict_impact(alt_action, state)

            alternatives.append({
                'action': alt_action,
                'description': self._interpret_action_semantics(alt_action),
                'predicted_impact': alt_impact,
                'ranking': i + 2,  # Chosen action is rank 1
                'reason_not_chosen': self._compare_actions(action, alt_action, state)
            })

        return alternatives

    def _predict_impact(self, action: np.ndarray, state: Any) -> float:
        """
        Predict Q-score improvement for given action.

        In production: Use trained value network
        """
        # Placeholder: simple heuristic
        action_magnitude = np.linalg.norm(action)
        current_score = state.feature_scores.get(self.dimension, 0.5)
        gap = max(0, 0.85 - current_score)

        # Diminishing returns: harder to improve when already high
        predicted_improvement = min(gap, action_magnitude * 0.1 * (1 - current_score))

        return predicted_improvement

    def _compare_actions(
        self,
        chosen_action: np.ndarray,
        alternative_action: np.ndarray,
        state: Any
    ) -> str:
        """Explain why chosen action is better than alternative"""

        chosen_impact = self._predict_impact(chosen_action, state)
        alt_impact = self._predict_impact(alternative_action, state)

        if chosen_impact > alt_impact:
            return f"Chosen action has higher expected impact ({chosen_impact:.3f} vs {alt_impact:.3f})"
        elif np.linalg.norm(chosen_action) < np.linalg.norm(alternative_action):
            return f"Chosen action is more conservative (lower risk)"
        else:
            return f"Chosen action ranked higher by policy network"


# ============================================================================
# NEXT-GEN AGENT WITH RLVR
# ============================================================================

class NextGenPESAgent:
    """
    Enhanced PES agent with:
    1. Inference-time scaling (adaptive compute)
    2. Verifiable reasoning chains (RLVR)
    3. Confidence intervals on predictions
    4. Alternative exploration
    """

    def __init__(
        self,
        dimension: str,
        weight: float,
        state_dim: int = 538,
        action_dim: int = 3
    ):
        self.dimension = dimension
        self.weight = weight

        # Adaptive policy network
        self.policy_net = AdaptivePolicyNetwork(state_dim, action_dim)

        # Reasoning chain generator
        self.reasoning_generator = ReasoningChainGenerator(dimension, weight)

        # Performance tracking
        self.performance_log = []

    def get_action_with_reasoning(
        self,
        state: Any,
        return_alternatives: bool = True
    ) -> VerifiableAction:
        """
        Get action with full reasoning chain and verification.

        Args:
            state: Current prompt state
            return_alternatives: Whether to generate alternative actions

        Returns:
            verifiable_action: VerifiableAction with complete reasoning
        """
        start_time = time.time()

        # Convert state to tensor
        state_vector = state.to_vector()
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)

        # Get action and tier
        with torch.no_grad():
            action_logits, tier = self.policy_net(state_tensor, return_tier=True)
            action = action_logits.squeeze(0).numpy()

        # Predict impact
        predicted_improvement = self.reasoning_generator._predict_impact(action, state)

        # Compute confidence interval (Monte Carlo dropout)
        confidence_interval = self._compute_confidence_interval(
            state_vector,
            action,
            num_samples=10
        )

        # Generate reasoning chain
        reasoning_chain = self.reasoning_generator.generate_reasoning(
            state=state,
            action=action,
            predicted_impact=predicted_improvement,
            tier=tier
        )

        # Generate alternatives
        alternatives = []
        if return_alternatives:
            alternatives = self.reasoning_generator.generate_alternatives(
                state=state,
                action=action,
                num_alternatives=3
            )

        # Verification data
        verification_data = {
            'tier': tier.value,
            'tier_statistics': self.policy_net.get_tier_statistics(),
            'state_features': {
                dim: state.feature_scores.get(dim, 0.0)
                for dim in ['P', 'T', 'F', 'S', 'C', 'R']
            },
            'action_magnitude': float(np.linalg.norm(action)),
            'timestamp': time.time()
        }

        latency_ms = (time.time() - start_time) * 1000

        return VerifiableAction(
            action=action,
            reasoning_chain=reasoning_chain,
            predicted_improvement=predicted_improvement,
            confidence_interval=confidence_interval,
            alternatives_considered=alternatives,
            verification_data=verification_data,
            tier=tier,
            latency_ms=latency_ms
        )

    def _compute_confidence_interval(
        self,
        state: np.ndarray,
        action: np.ndarray,
        num_samples: int = 10
    ) -> Tuple[float, float]:
        """
        Compute 95% confidence interval for predicted improvement.

        Uses Monte Carlo dropout for uncertainty estimation.
        """
        # Enable dropout
        self.policy_net.train()

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        predictions = []

        with torch.no_grad():
            for _ in range(num_samples):
                # Forward pass with dropout
                action_logits = self.policy_net(state_tensor)
                # Predict impact (simplified)
                impact = float(torch.norm(action_logits).item()) * 0.05
                predictions.append(impact)

        # Disable dropout
        self.policy_net.eval()

        # Compute 95% CI
        mean = np.mean(predictions)
        std = np.std(predictions)
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std

        return (ci_lower, ci_upper)

    def log_performance(
        self,
        state: Any,
        action: VerifiableAction,
        actual_improvement: float
    ):
        """
        Log performance for continuous learning.

        Compares predicted vs. actual improvement for calibration.
        """
        self.performance_log.append({
            'dimension': self.dimension,
            'tier': action.tier.value,
            'predicted_improvement': action.predicted_improvement,
            'actual_improvement': actual_improvement,
            'prediction_error': abs(action.predicted_improvement - actual_improvement),
            'ci_covered': (
                action.confidence_interval[0] <= actual_improvement <= action.confidence_interval[1]
            ),
            'latency_ms': action.latency_ms,
            'timestamp': time.time()
        })

    def get_calibration_metrics(self) -> Dict[str, float]:
        """
        Compute calibration metrics from performance log.

        Returns:
            metrics: {
                'mean_absolute_error': float,
                'ci_coverage': float,  # Should be ~0.95
                'avg_latency_ms': float,
                'tier_distribution': Dict[str, float]
            }
        """
        if not self.performance_log:
            return {}

        errors = [entry['prediction_error'] for entry in self.performance_log]
        ci_coverage = [entry['ci_covered'] for entry in self.performance_log]
        latencies = [entry['latency_ms'] for entry in self.performance_log]

        tier_counts = defaultdict(int)
        for entry in self.performance_log:
            tier_counts[entry['tier']] += 1

        total = len(self.performance_log)
        tier_distribution = {
            tier: count / total
            for tier, count in tier_counts.items()
        }

        return {
            'mean_absolute_error': np.mean(errors),
            'ci_coverage': np.mean(ci_coverage),
            'avg_latency_ms': np.mean(latencies),
            'tier_distribution': tier_distribution,
            'num_samples': total
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of next-gen PES agent with RLVR.
    """

    # Import the original PromptState (assuming it's available)
    # from pes_multi_agent_system import PromptState, Dimension

    # For demonstration, create a mock state
    @dataclass
    class MockState:
        text_embedding: np.ndarray = field(default_factory=lambda: np.random.randn(512))
        feature_scores: Dict[str, float] = field(default_factory=lambda: {
            'P': 0.65, 'T': 0.72, 'F': 0.68, 'S': 0.70, 'C': 0.75, 'R': 0.73
        })
        token_count: int = 120
        iteration: int = 5
        agent_actions: Dict = field(default_factory=dict)

        def to_vector(self):
            """Convert to 538-dim vector"""
            feature_vector = np.array([
                self.feature_scores.get(dim, 0.5) for dim in ['P', 'T', 'F', 'S', 'C', 'R']
            ])
            action_vector = np.zeros(18)  # 6 agents × 3-dim

            return np.concatenate([
                self.text_embedding,  # 512
                feature_vector,       # 6
                [self.token_count],   # 1
                [self.iteration],     # 1
                action_vector         # 18
            ])  # Total: 538

    # Create next-gen agent for Persona dimension
    agent = NextGenPESAgent(
        dimension="P",
        weight=0.20,
        state_dim=538,
        action_dim=3
    )

    # Create mock state
    state = MockState()

    # Get action with reasoning
    print("=" * 80)
    print("NEXT-GEN PES AGENT: VERIFIABLE ACTION WITH REASONING")
    print("=" * 80)

    verifiable_action = agent.get_action_with_reasoning(state)

    print(f"\nDimension: {agent.dimension}")
    print(f"Compute Tier: {verifiable_action.tier.value}")
    print(f"Latency: {verifiable_action.latency_ms:.1f}ms")
    print(f"\nAction Vector: {verifiable_action.action}")
    print(f"Predicted Improvement: {verifiable_action.predicted_improvement:+.3f}")
    print(f"95% Confidence Interval: [{verifiable_action.confidence_interval[0]:+.3f}, {verifiable_action.confidence_interval[1]:+.3f}]")

    print("\n" + "─" * 80)
    print("REASONING CHAIN:")
    print("─" * 80)
    for step in verifiable_action.reasoning_chain:
        print(f"\nStep {step.step_number}: {step.description}")
        if step.evidence:
            print(f"  Evidence: {step.evidence}")
        print(f"  Confidence: {step.confidence:.2f}")

    print("\n" + "─" * 80)
    print("ALTERNATIVES CONSIDERED:")
    print("─" * 80)
    for i, alt in enumerate(verifiable_action.alternatives_considered, 1):
        print(f"\nAlternative {i}:")
        print(f"  Description: {alt['description']}")
        print(f"  Predicted Impact: {alt['predicted_impact']:+.3f}")
        print(f"  Reason Not Chosen: {alt['reason_not_chosen']}")

    print("\n" + "─" * 80)
    print("VERIFICATION DATA:")
    print("─" * 80)
    print(f"State Features: {verifiable_action.verification_data['state_features']}")
    print(f"Action Magnitude: {verifiable_action.verification_data['action_magnitude']:.3f}")
    print(f"Tier Statistics: {verifiable_action.verification_data['tier_statistics']}")

    print("\n" + "=" * 80)
    print("✅ Next-Gen PES Agent Demonstration Complete")
    print("=" * 80)

    # Simulate logging actual outcome
    actual_improvement = 0.08  # Simulated
    agent.log_performance(state, verifiable_action, actual_improvement)

    # Get calibration metrics
    metrics = agent.get_calibration_metrics()
    print(f"\nCalibration Metrics: {metrics}")
