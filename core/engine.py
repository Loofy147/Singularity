"""
REALIZATION ENGINE (V2)
=======================
The foundational system for managing crystallized knowledge.
"""

import json
import re
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib


@dataclass
class QualityDimension:
    """A single dimension of quality used to calculate Q-score."""
    name: str
    description: str
    weight: float
    discovered_by: str = "human"  # 'human' or 'singularity'

    def __str__(self):
        return f"{self.name} (w={self.weight:.2f}) - {self.description}"


@dataclass
class RealizationFeatures:
    """
    Quality features of a realization.
    Supports both core dimensions and emergent ones.
    """
    scores: Dict[str, float] = field(default_factory=dict)

    def validate(self):
        """Ensure all scores are in [0, 1] range."""
        for name, value in self.scores.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Feature '{name}' must be between 0 and 1, got {value}")

    @classmethod
    def from_core(cls, grounding: float, certainty: float, structure: float,
                  applicability: float, coherence: float, generativity: float):
        return cls(scores={
            'grounding': grounding,
            'certainty': certainty,
            'structure': structure,
            'applicability': applicability,
            'coherence': coherence,
            'generativity': generativity
        })


@dataclass
class Realization:
    """A crystallized unit of knowledge."""
    id: str
    content: str
    features: RealizationFeatures
    q_score: float
    layer: Any  # int or 'N'
    timestamp: str
    parents: List[str]
    children: List[str]
    turn_number: int
    context: str = ""
    evidence: List[str] = field(default_factory=list)


class RealizationEngine:
    """
    Advanced engine for managing the realization lifecycle.
    """

    CORE_DIMENSIONS = {
        'grounding': QualityDimension('Grounding', 'Rootedness in facts/rules', 0.18),
        'certainty': QualityDimension('Certainty', 'Self-certifying precision', 0.22),
        'structure': QualityDimension('Structure', 'Crystallization clarity', 0.20),
        'applicability': QualityDimension('Applicability', 'Actionability/usefulness', 0.18),
        'coherence': QualityDimension('Coherence', 'Consistency with prior knowledge', 0.12),
        'generativity': QualityDimension('Generativity', 'Daughter idea potential', 0.10)
    }

    def __init__(self, dimensions: Optional[Dict[str, QualityDimension]] = None):
        self.dimensions = dimensions or self.CORE_DIMENSIONS.copy()
        self.layers = {0: {}, 1: {}, 2: {}, 3: {}, 'N': {}}
        self.index: Dict[str, Realization] = {}
        self.stats = {
            'total_realizations': 0,
            'layer_distribution': {0: 0, 1: 0, 2: 0, 3: 0, 'N': 0},
            'avg_q_score': 0.0
        }

    def calculate_q_score(self, features: RealizationFeatures) -> Tuple[float, str]:
        features.validate()
        total_weight = sum(d.weight for d in self.dimensions.values())
        norm = 1.0 / total_weight if abs(total_weight - 1.0) > 0.001 else 1.0
        total_q = 0.0
        calc_parts = []
        for key, dim in self.dimensions.items():
            score = features.scores.get(key, 0.5)
            contribution = dim.weight * score * norm
            total_q += contribution
            calc_parts.append(f"{dim.weight:.2f}x{score:.2f}")
        return round(total_q, 4), " + ".join(calc_parts)

    def assign_layer(self, q_score: float, features: RealizationFeatures) -> Any:
        grounding = features.scores.get('grounding', 0.0)
        if q_score >= 0.95 and grounding >= 0.90: return 0
        if q_score >= 0.92 and grounding >= 0.80: return 1
        if q_score >= 0.85 and grounding >= 0.70: return 2
        if q_score >= 0.75 and grounding >= 0.50: return 3
        return 'N'

    def add_realization(self, content: str, features: RealizationFeatures,
                        turn_number: int, parents: List[str] = None,
                        context: str = "", evidence: List[str] = None) -> Realization:
        parents = parents or []
        evidence = evidence or []
        q_score, _ = self.calculate_q_score(features)
        layer = self.assign_layer(q_score, features)
        r_id = f"R_{hashlib.sha256(content.encode()).hexdigest()[:8]}"
        r = Realization(id=r_id, content=content, features=features, q_score=q_score,
                       layer=layer, timestamp=datetime.now().isoformat(),
                       parents=parents, children=[], turn_number=turn_number,
                       context=context, evidence=evidence)
        self.layers[layer][r_id] = r
        self.index[r_id] = r
        for p_id in parents:
            if p_id in self.index and r_id not in self.index[p_id].children:
                self.index[p_id].children.append(r_id)
        self._update_stats(r)
        return r

    def _update_stats(self, r: Realization):
        self.stats['total_realizations'] += 1
        self.stats['layer_distribution'][r.layer] += 1
        total_q = sum(res.q_score for res in self.index.values())
        self.stats['avg_q_score'] = total_q / self.stats['total_realizations']

    def print_stats(self):
        print("\n" + "="*60)
        print("REALIZATION ENGINE STATISTICS")
        print("="*60)
        print(f"Total Realizations: {self.stats['total_realizations']}")
        print(f"Average Q-Score: {self.stats['avg_q_score']:.4f}")
        print("\nLayer Distribution:")
        for layer in [0, 1, 2, 3, 'N']:
            count = self.stats['layer_distribution'][layer]
            pct = (count / self.stats['total_realizations'] * 100) if self.stats['total_realizations'] > 0 else 0
            name = {0:"Universal", 1:"Domain Facts", 2:"Patterns", 3:"Situational", 'N':"Ephemeral"}[layer]
            print(f"  Layer {layer} ({name}): {count} ({pct:.1f}%)")
        print("="*60 + "\n")

    def retrieve(self, query: str, limit: int = 5) -> List[Realization]:
        query_terms = set(re.findall(r'\w+', query.lower()))
        scores = []
        for r in self.index.values():
            overlap = len(query_terms & set(re.findall(r'\w+', r.content.lower())))
            if overlap > 0:
                scores.append(( (overlap/len(query_terms)) * r.q_score, r))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [r for score, r in scores[:limit]]

    def export_json(self, path: str):
        data = {
            'stats': self.stats,
            'dimensions': {k: asdict(v) for k, v in self.dimensions.items()},
            'realizations': [asdict(r) for r in self.index.values()]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    engine = RealizationEngine()
    print("Engine initialized.")
