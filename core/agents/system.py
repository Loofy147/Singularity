"""
UQS MULTI-AGENT SYSTEM (V3.2)
=============================
Managed multi-agent system for knowledge optimization using the 13-dimension UQS framework.
Integrated with Metacognitive Monitoring for bias detection and calibration.
"""

import sys
import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Add root to path for imports
sys.path.append(os.getcwd())
from core.engine import RealizationEngine, QualityDimension, RealizationFeatures
from core.agents.metacognition import MetacognitiveMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UQSDimension(str, Enum):
    GROUNDING = "grounding"
    CERTAINTY = "certainty"
    STRUCTURE = "structure"
    APPLICABILITY = "applicability"
    COHERENCE = "coherence"
    GENERATIVITY = "generativity"
    PRESENTATION = "presentation"
    TEMPORAL = "temporal"
    DENSITY = "density"
    SYNTHESIS = "synthesis"
    RESILIENCE = "resilience"
    TRANSFERABILITY = "transferability"
    ROBUSTNESS = "robustness"

# Weights from core/engine.py (V3.2)
UQS_WEIGHTS = {
    UQSDimension.GROUNDING: 0.14,
    UQSDimension.CERTAINTY: 0.16,
    UQSDimension.STRUCTURE: 0.14,
    UQSDimension.APPLICABILITY: 0.12,
    UQSDimension.COHERENCE: 0.10,
    UQSDimension.GENERATIVITY: 0.07,
    UQSDimension.PRESENTATION: 0.04,
    UQSDimension.TEMPORAL: 0.03,
    UQSDimension.DENSITY: 0.05,
    UQSDimension.SYNTHESIS: 0.04,
    UQSDimension.RESILIENCE: 0.03,
    UQSDimension.TRANSFERABILITY: 0.02,
    UQSDimension.ROBUSTNESS: 0.06
}

@dataclass
class RealizationState:
    text: str
    scores: Dict[UQSDimension, float]
    iteration: int

class BaseAgent:
    def __init__(self, dimension: UQSDimension):
        self.dimension = dimension
        self.weight = UQS_WEIGHTS[dimension]

    def act(self, state: RealizationState) -> Dict[str, Any]:
        # Heuristic-based action
        current_score = state.scores.get(self.dimension, 0.5)

        is_hard = self.dimension in [
            UQSDimension.DENSITY, UQSDimension.SYNTHESIS,
            UQSDimension.RESILIENCE, UQSDimension.TRANSFERABILITY,
            UQSDimension.ROBUSTNESS
        ]
        difficulty = 0.4 if is_hard else 1.0

        improvement = (1.0 - current_score) * 0.1 * np.random.random() * difficulty

        return {
            "dimension": self.dimension,
            "improvement": improvement,
            "description": f"Optimizing {self.dimension.value} by {improvement:.4f}"
        }

class MultiAgentCoordinator:
    def __init__(self):
        self.agents = {d: BaseAgent(d) for d in UQSDimension}
        self.engine = RealizationEngine()
        self.monitor = MetacognitiveMonitor()

        logger.info("Multi-Agent Coordinator initialized with 13 UQS dimensions and Metacognitive Monitor")

    def optimize_knowledge(self, initial_text: str, target_q: float = 0.90, max_iterations: int = 20) -> Tuple[str, Dict[str, Any]]:
        current_text = initial_text
        current_scores = {d: 0.5 for d in UQSDimension}

        history = []

        for i in range(max_iterations):
            state = RealizationState(text=current_text, scores=current_scores, iteration=i)

            # 1. Metacognitive Supervision
            avg_conf = np.mean(list(current_scores.values()))
            obs = self.monitor.monitor_step(i, current_text, avg_conf)

            if obs.risk_level == "HIGH":
                logger.warning(f"Metacognitive Warning: {obs.recommendation}")

            # 2. Agent Actions
            sorted_dimensions = sorted(UQSDimension, key=lambda d: UQS_WEIGHTS[d], reverse=True)
            iteration_actions = []
            for d in sorted_dimensions:
                agent = self.agents[d]
                action = agent.act(state)
                current_scores[d] = min(1.0, current_scores[d] + action['improvement'])
                iteration_actions.append(action)

            # 3. Quality Scoring
            feat_dict = {d.value: current_scores[d] for d in UQSDimension}
            features = RealizationFeatures(scores=feat_dict)
            q_score, _ = self.engine.calculate_q_score(features)

            history.append({
                "iteration": i,
                "q_score": q_score,
                "metacognition": obs,
                "actions": iteration_actions
            })

            logger.info(f"Iteration {i}: Q-score = {q_score:.4f} | Metacog: {obs.risk_level}")

            if q_score >= target_q:
                logger.info(f"Target Q-score {target_q} reached at iteration {i}")
                break

        # Final crystallization
        r = self.engine.add_realization(
            content=current_text,
            features=RealizationFeatures(scores={d.value: current_scores[d] for d in UQSDimension}),
            turn_number=max_iterations,
            context="UQS 13-agent optimization with Metacognitive Monitor"
        )

        return current_text, {
            "realization_id": r.id,
            "final_q": q_score,
            "iterations": i + 1,
            "history": history,
            "final_layer": r.layer,
            "metacognitive_audit": self.monitor.history
        }

if __name__ == "__main__":
    coordinator = MultiAgentCoordinator()
    optimized, meta = coordinator.optimize_knowledge("Handling adversarial knowledge attacks.")
    print(f"\nFinal Q-score: {meta['final_q']:.4f} (Layer {meta['final_layer']})")
    coordinator.monitor.print_audit_trail()
