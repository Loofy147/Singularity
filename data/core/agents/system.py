"""
PES MULTI-AGENT SYSTEM
======================
Managed multi-agent system for prompt optimization using the PES framework.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dimension(str, Enum):
    PERSONA = "P"
    TONE = "T"
    FORMAT = "F"
    SPECIFICITY = "S"
    CONSTRAINTS = "C"
    CONTEXT = "R"

PES_WEIGHTS = {
    Dimension.PERSONA: 0.20,
    Dimension.TONE: 0.18,
    Dimension.FORMAT: 0.18,
    Dimension.SPECIFICITY: 0.18,
    Dimension.CONSTRAINTS: 0.13,
    Dimension.CONTEXT: 0.13
}

@dataclass
class PromptState:
    text: str
    scores: Dict[Dimension, float]
    iteration: int

class BaseAgent:
    def __init__(self, dimension: Dimension):
        self.dimension = dimension
        self.weight = PES_WEIGHTS[dimension]

    def act(self, state: PromptState) -> Dict[str, Any]:
        # Simple heuristic-based action for demonstration
        # In a real RL system, this would use a policy network
        current_score = state.scores.get(self.dimension, 0.5)
        improvement = (1.0 - current_score) * 0.1 * np.random.random()
        return {
            "dimension": self.dimension,
            "improvement": improvement,
            "description": f"Improving {self.dimension.name} by {improvement:.4f}"
        }

class MultiAgentCoordinator:
    def __init__(self):
        self.agents = {d: BaseAgent(d) for d in Dimension}

        # Initialize an engine specifically for prompts
        prompt_dims = {
            d.name.lower(): QualityDimension(d.name, f"PES {d.name} dimension", PES_WEIGHTS[d])
            for d in Dimension
        }
        self.engine = RealizationEngine(dimensions=prompt_dims)

        logger.info("Multi-Agent Coordinator initialized with PES dimensions")

    def optimize_prompt(self, initial_prompt: str, target_q: float = 0.85, max_iterations: int = 10) -> Tuple[str, Dict[str, Any]]:
        current_text = initial_prompt
        current_scores = {d: 0.5 for d in Dimension}

        history = []

        for i in range(max_iterations):
            state = PromptState(text=current_text, scores=current_scores, iteration=i)

            # Agents act in order of weight (highest priority first)
            sorted_dimensions = sorted(Dimension, key=lambda d: PES_WEIGHTS[d], reverse=True)

            iteration_actions = []
            for d in sorted_dimensions:
                agent = self.agents[d]
                action = agent.act(state)

                # Apply action (simulated)
                current_scores[d] = min(1.0, current_scores[d] + action['improvement'])
                iteration_actions.append(action)

            # Calculate new Q-score using the engine
            feat_dict = {d.name.lower(): current_scores[d] for d in Dimension}
            features = RealizationFeatures(scores=feat_dict)
            q_score, _ = self.engine.calculate_q_score(features)

            history.append({
                "iteration": i,
                "q_score": q_score,
                "actions": iteration_actions
            })

            logger.info(f"Iteration {i}: Q-score = {q_score:.4f}")

            if q_score >= target_q:
                logger.info(f"Target Q-score reached at iteration {i}")
                break

        # Final crystallization of the optimized prompt as a realization
        self.engine.add_realization(
            content=current_text,
            features=RealizationFeatures(scores={d.name.lower(): current_scores[d] for d in Dimension}),
            turn_number=max_iterations,
            context="Multi-agent optimization"
        )

        return current_text, {
            "final_q": q_score,
            "iterations": i + 1,
            "history": history
        }

if __name__ == "__main__":
    coordinator = MultiAgentCoordinator()
    optimized, meta = coordinator.optimize_prompt("Write a poem about space.")
    print(f"\nFinal Q-score: {meta['final_q']:.4f}")
    print(f"Iterations: {meta['iterations']}")
