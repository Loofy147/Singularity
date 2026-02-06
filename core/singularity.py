"""
SINGULARITY REALIZATION ENGINE
==============================
Meta-framework that evolves the quality dimensions of the Realization Engine.
"""

import sys
import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Add root to path
sys.path.append(os.getcwd())
from core.engine import RealizationEngine, RealizationFeatures, QualityDimension, Realization

class SingularityRealizationEngine:
    """
    Evolves the RealizationEngine by adjusting weights and discovering new dimensions.
    """

    def __init__(self, base_engine: RealizationEngine):
        self.base_engine = base_engine
        self.evolution_history = []
        self.performance_history = []

        # Adaptation parameters
        self.weight_adaptation_rate = 0.05
        self.convergence_threshold = 0.001
        self.discovered_count = 0
        self.momentum = 0.9  # Reduced momentum for faster testing

    def evolve(self, realizations: List[Realization], q_scores: List[float]) -> Dict[str, Any]:
        """
        Analyze recent performance and evolve the framework.
        """
        # 1. Discovery: Look for latent patterns that suggest new dimensions
        new_dims = self._discover_latent_dimensions(realizations)

        # 2. Adaptation: Adjust weights of existing dimensions
        weight_updates = self._compute_weight_updates(realizations, q_scores)

        # Apply updates to base engine
        for key, new_weight in weight_updates.items():
            if key in self.base_engine.dimensions:
                self.base_engine.dimensions[key].weight = float(new_weight)

        for dim in new_dims:
            self.base_engine.dimensions[dim.name.lower().replace(' ', '_')] = dim
            self.discovered_count += 1

        # Record evolution
        evolution_record = {
            'timestamp': time.time(),
            'avg_q_score': float(np.mean(q_scores)),
            'discovered_dimensions': [d.name for d in new_dims],
            'weight_updates': {k: float(v) for k, v in weight_updates.items()},
            'current_weights': {k: float(v.weight) for k, v in self.base_engine.dimensions.items()}
        }
        self.evolution_history.append(evolution_record)

        return evolution_record

    def _discover_latent_dimensions(self, realizations: List[Realization]) -> List[QualityDimension]:
        discovered = []
        if len(realizations) > 20 and self.discovered_count < 3:
            has_high_syn = any(r.features.scores.get('synthesis', 0) > 0.9 for r in realizations)
            if has_high_syn and 'metacognitive' not in [d.name.lower() for d in self.base_engine.dimensions.values()]:
                discovered.append(QualityDimension(
                    name="Metacognitive Awareness",
                    description="System's ability to self-monitor and correct reasoning paths",
                    weight=0.04,
                    discovered_by="singularity"
                ))
        return discovered

    def _compute_weight_updates(self, realizations: List[Realization], q_scores: List[float]) -> Dict[str, float]:
        # Focus on the top 25% of realizations
        if not q_scores: return {k: v.weight for k, v in self.base_engine.dimensions.items()}
        threshold = np.percentile(q_scores, 75)
        high_quality = [r for r in realizations if r.q_score >= threshold]

        if not high_quality:
            return {k: v.weight for k, v in self.base_engine.dimensions.items()}

        new_weights = {}
        dim_importance = {}

        for key in self.base_engine.dimensions:
            avg_score = np.mean([r.features.scores.get(key, 0.5) for r in high_quality])
            dim_importance[key] = avg_score * self.base_engine.dimensions[key].weight

        total_importance = sum(dim_importance.values())
        if total_importance == 0:
            return {k: v.weight for k, v in self.base_engine.dimensions.items()}

        for key, dim in self.base_engine.dimensions.items():
            target_weight = dim_importance[key] / total_importance
            new_weights[key] = (dim.weight * self.momentum) + (target_weight * (1 - self.momentum))

        norm_sum = sum(new_weights.values())
        for key in new_weights:
            new_weights[key] /= norm_sum

        return new_weights

    def print_status(self):
        print(f"\n🌌 SINGULARITY ENGINE STATUS")
        print(f"Dimensions: {len(self.base_engine.dimensions)}")
        print(f"Discovered: {self.discovered_count}")
        if self.evolution_history:
            latest = self.evolution_history[-1]
            print(f"Latest Evolution: {time.ctime(latest['timestamp'])}")
            print(f"Avg Q-Score: {latest['avg_q_score']:.4f}")
            print("Top weights:")
            sorted_weights = sorted(latest['current_weights'].items(), key=lambda x: x[1], reverse=True)
            for k, v in sorted_weights[:5]:
                print(f"  - {k}: {v:.4f}")

if __name__ == "__main__":
    from core.engine import RealizationEngine
    base = RealizationEngine()
    sre = SingularityRealizationEngine(base)
    mock_realizations = []
    for i in range(25):
        val = 0.9 if i < 10 else 0.6
        f = RealizationFeatures.from_uqs(val, val, val, val, val, val, val, val)
        r = base.add_realization(f"Sample {i}", f, 1)
        mock_realizations.append(r)
    sre.evolve(mock_realizations, [r.q_score for r in mock_realizations])
    sre.print_status()
