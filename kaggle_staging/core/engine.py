"""
REALIZATION ENGINE (V3.1) - EMERGENT UQS EDITION
================================================
Evolved with 12 dimensions including predicted emergents:
D7 (Density), D8 (Synthesis), D9 (Resilience), D10 (Transferability).
"""

import json
import re
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib


class RealizationFeatures:
    """
    Quality features of a realization.
    Supports 12-dimension UQS including emergents.
    """
    def __init__(self, scores: Optional[Dict[str, float]] = None, **kwargs):
        self.scores = scores or {}
        self.scores.update(kwargs)
        # Ensure all 12 dims have a value
        dimensions = [
            'grounding', 'certainty', 'structure', 'applicability',
            'coherence', 'generativity', 'presentation', 'temporal',
            'density', 'synthesis', 'resilience', 'transferability'
        ]
        for dim in dimensions:
            if dim not in self.scores:
                self.scores[dim] = 0.5

    @property
    def grounding(self): return self.scores.get('grounding', 0.5)
    @property
    def certainty(self): return self.scores.get('certainty', 0.5)
    @property
    def structure(self): return self.scores.get('structure', 0.5)
    @property
    def applicability(self): return self.scores.get('applicability', 0.5)
    @property
    def coherence(self): return self.scores.get('coherence', 0.5)
    @property
    def generativity(self): return self.scores.get('generativity', 0.5)
    @property
    def presentation(self): return self.scores.get('presentation', 0.5)
    @property
    def temporal(self): return self.scores.get('temporal', 0.5)
    @property
    def density(self): return self.scores.get('density', 0.5)
    @property
    def synthesis(self): return self.scores.get('synthesis', 0.5)
    @property
    def resilience(self): return self.scores.get('resilience', 0.5)
    @property
    def transferability(self): return self.scores.get('transferability', 0.5)

    def validate(self):
        """Ensure all scores are in [0, 1] range."""
        for name, value in self.scores.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Feature '{name}' must be between 0 and 1, got {value}")

    @classmethod
    def from_uqs(cls, grounding: float, certainty: float, structure: float,
                  applicability: float, coherence: float, generativity: float,
                  presentation: float, temporal: float, density: float = 0.5,
                  synthesis: float = 0.5, resilience: float = 0.5,
                  transferability: float = 0.5):
        return cls(scores={
            'grounding': grounding,
            'certainty': certainty,
            'structure': structure,
            'applicability': applicability,
            'coherence': coherence,
            'generativity': generativity,
            'presentation': presentation,
            'temporal': temporal,
            'density': density,
            'synthesis': synthesis,
            'resilience': resilience,
            'transferability': transferability
        })

    @classmethod
    def from_core(cls, grounding: float, certainty: float, structure: float,
                  applicability: float, coherence: float, generativity: float):
        """Legacy method for backward compatibility."""
        return cls.from_uqs(grounding, certainty, structure, applicability,
                             coherence, generativity, 0.5, 0.5)

    def to_dict(self):
        return {'scores': self.scores}


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
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_number: int
    description: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ReasoningChain:
    """A complete chain of reasoning behind a realization."""
    steps: List[ReasoningStep] = field(default_factory=list)
    total_confidence: float = 1.0


@dataclass
class Relation:
    """A directed, typed relationship between realizations."""
    target_id: str
    type: str  # 'derivation', 'synthesis', 'contradiction', 'refinement'
    strength: float = 1.0


@dataclass
class Realization:
    """A crystallized unit of knowledge with reasoning and topology."""
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
    reasoning_chain: Optional[ReasoningChain] = None
    topology_relations: List[Relation] = field(default_factory=list)

    def to_dict(self):
        res = asdict(self)
        res['features'] = self.features.to_dict()
        return res


class RealizationEngine:
    """
    Advanced engine implementing the Universal Quality Score (UQS) with emergents.
    """

    UQS_DIMENSIONS = {
        'grounding': QualityDimension('Grounding/Persona', 'Rootedness in facts/rules', 0.15),
        'certainty': QualityDimension('Certainty', 'Self-certifying precision', 0.18),
        'structure': QualityDimension('Structure/Specificity', 'Crystallization clarity', 0.15),
        'applicability': QualityDimension('Applicability', 'Actionability/usefulness', 0.14),
        'coherence': QualityDimension('Coherence/Context', 'Consistency with prior knowledge', 0.10),
        'generativity': QualityDimension('Generativity', 'Daughter idea potential', 0.07),
        'presentation': QualityDimension('Presentation', 'Format and tone quality', 0.04),
        'temporal': QualityDimension('Temporal', 'Resilience over time', 0.03),
        'density': QualityDimension('بنات افكار Density', 'Rate of spawning daughters', 0.05, discovered_by='singularity'),
        'synthesis': QualityDimension('Convergence Synthesis', 'Multi-parent integration quality', 0.04, discovered_by='singularity'),
        'resilience': QualityDimension('Temporal Resilience', 'Stability across paradigm shifts', 0.03, discovered_by='singularity'),
        'transferability': QualityDimension('Cross-Domain Transferability', 'Applicability across fields', 0.02, discovered_by='singularity')
    }

    def __init__(self, dimensions: Optional[Dict[str, QualityDimension]] = None):
        self.dimensions = dimensions or self.UQS_DIMENSIONS.copy()
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
        grounding = features.grounding
        # Thresholds from Evolved Framework Guide
        if q_score >= 0.95 and grounding >= 0.90: return 0
        if q_score >= 0.92 and grounding >= 0.80: return 1
        if q_score >= 0.85 and grounding >= 0.70: return 2
        if q_score >= 0.75 and grounding >= 0.50: return 3
        return 'N'

    def add_realization(self, content: str, features: RealizationFeatures,
                        turn_number: int, parents: List[str] = None,
                        context: str = "", evidence: List[str] = None,
                        reasoning_chain: Optional[ReasoningChain] = None,
                        topology_relations: List[Relation] = None) -> Realization:
        parents = parents or []
        evidence = evidence or []
        topology_relations = topology_relations or []

        q_score, _ = self.calculate_q_score(features)
        layer = self.assign_layer(q_score, features)

        r_id = f"R_{hashlib.sha256(content.encode()).hexdigest()[:8]}"
        r = Realization(id=r_id, content=content, features=features, q_score=q_score,
                       layer=layer, timestamp=datetime.now().isoformat(),
                       parents=parents, children=[], turn_number=turn_number,
                       context=context, evidence=evidence,
                       reasoning_chain=reasoning_chain,
                       topology_relations=topology_relations)

        self.layers[layer][r_id] = r
        self.index[r_id] = r

        # Link parents and children
        for p_id in parents:
            if p_id in self.index:
                if r_id not in self.index[p_id].children:
                    self.index[p_id].children.append(r_id)
                # Auto-create derivation relation if not exists
                if not any(rel.target_id == r_id for rel in self.index[p_id].topology_relations):
                    self.index[p_id].topology_relations.append(Relation(target_id=r_id, type='derivation'))

        self._update_stats(r)
        return r

    def _update_stats(self, r: Realization):
        self.stats['total_realizations'] += 1
        self.stats['layer_distribution'][r.layer] += 1
        total_q = sum(res.q_score for res in self.index.values())
        self.stats['avg_q_score'] = total_q / self.stats['total_realizations']

    def print_stats(self):
        print("\n" + "="*60)
        print("EMERGENT UQS REALIZATION ENGINE STATISTICS")
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

    def get_family_tree(self, r_id: str, depth: int = 2) -> Dict[str, Any]:
        """Retrieve the topology tree for a realization."""
        if r_id not in self.index or depth < 0:
            return {}

        r = self.index[r_id]
        return {
            'id': r.id,
            'content': r.content[:50] + "...",
            'q_score': r.q_score,
            'layer': r.layer,
            'relations': [
                {
                    'target_id': rel.target_id,
                    'type': rel.type,
                    'tree': self.get_family_tree(rel.target_id, depth - 1)
                }
                for rel in r.topology_relations
            ]
        }

    def export_json(self, path: str):
        data = {
            'stats': self.stats,
            'dimensions': {k: asdict(v) for k, v in self.dimensions.items()},
            'realizations': [r.to_dict() for r in self.index.values()]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    engine = RealizationEngine()
    print("Emergent UQS Engine initialized.")
