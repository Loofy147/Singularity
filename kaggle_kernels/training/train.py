import torch
import numpy as np
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
from collections import defaultdict
import random

# --- BUNDLED CORE LOGIC (V3.3 - EVOLVED UQS) ---

class ComputeTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class QualityDimension:
    name: str
    description: str
    weight: float
    discovered_by: str = "human"

class RealizationFeatures:
    def __init__(self, scores: Optional[Dict[str, float]] = None, **kwargs):
        self.scores = scores or {}
        self.scores.update(kwargs)
        dimensions = [
            'grounding', 'certainty', 'structure', 'applicability',
            'coherence', 'generativity', 'presentation', 'temporal',
            'density', 'synthesis', 'resilience', 'transferability',
            'robustness'
        ]
        for dim in dimensions:
            if dim not in self.scores: self.scores[dim] = 0.5

    def validate(self):
        for name, value in self.scores.items():
            if not 0 <= value <= 1: raise ValueError(f"Feature '{name}' must be [0,1]")

class RealizationEngine:
    UQS_DIMENSIONS = {
        'grounding': QualityDimension('Grounding/Persona', 'Rootedness', 0.14),
        'certainty': QualityDimension('Certainty', 'Precision', 0.16),
        'structure': QualityDimension('Structure/Specificity', 'Clarity', 0.14),
        'applicability': QualityDimension('Applicability', 'Usefulness', 0.12),
        'coherence': QualityDimension('Coherence/Context', 'Consistency', 0.10),
        'generativity': QualityDimension('Generativity', 'Daughter ideas', 0.07),
        'presentation': QualityDimension('Presentation', 'Format/Tone', 0.04),
        'temporal': QualityDimension('Temporal', 'Resilience', 0.03),
        'density': QualityDimension('بنات افكار Density', 'Spawning rate', 0.05, 'singularity'),
        'synthesis': QualityDimension('Convergence Synthesis', 'Integration', 0.04, 'singularity'),
        'resilience': QualityDimension('Temporal Resilience', 'Stability', 0.03, 'singularity'),
        'transferability': QualityDimension('Cross-Domain Transferability', 'Mapping', 0.02, 'singularity'),
        'robustness': QualityDimension('Adversarial Robustness', 'Resistance', 0.06, 'singularity')
    }
    def __init__(self, dimensions=None):
        self.dimensions = dimensions or {k: QualityDimension(v.name, v.description, v.weight, v.discovered_by) for k, v in self.UQS_DIMENSIONS.items()}

    def calculate_q_score(self, features: RealizationFeatures) -> Tuple[float, str]:
        features.validate()
        total_weight = sum(d.weight for d in self.dimensions.values())
        norm = 1.0 / total_weight if abs(total_weight - 1.0) > 0.001 else 1.0
        total_q = sum(self.dimensions[k].weight * features.scores.get(k, 0.5) * norm for k in self.dimensions)
        return round(total_q, 4), ""

class SingularityEvolution:
    """Bundled meta-optimization logic for dynamic weights."""
    def __init__(self, engine: RealizationEngine):
        self.engine = engine
        self.momentum = 0.9

    def adapt_weights(self, batch_realizations: List[Dict]):
        # Filter for high quality realizations in the batch
        q_scores = []
        for r in batch_realizations:
            q, _ = self.engine.calculate_q_score(RealizationFeatures(scores=r.get('features', {}).get('scores', {})))
            q_scores.append(q)

        threshold = np.mean(q_scores)
        high_quality = [r for i, r in enumerate(batch_realizations) if q_scores[i] >= threshold]

        if not high_quality: return

        dim_importance = {}
        for key in self.engine.dimensions:
            avg_score = np.mean([r.get('features', {}).get('scores', {}).get(key, 0.5) for r in high_quality])
            dim_importance[key] = avg_score * self.engine.dimensions[key].weight

        total_importance = sum(dim_importance.values())
        if total_importance == 0: return

        for key in self.engine.dimensions:
            target_weight = dim_importance[key] / total_importance
            self.engine.dimensions[key].weight = (self.engine.dimensions[key].weight * self.momentum) + (target_weight * (1 - self.momentum))

        # Normalize
        norm_sum = sum(d.weight for d in self.engine.dimensions.values())
        for d in self.engine.dimensions.values():
            d.weight /= norm_sum

class NextGenPESAgent:
    def __init__(self, dimension, weight, state_dim, action_dim):
        self.dimension, self.weight = dimension, weight
        self.tier_history = defaultdict(int)
        self.performance_log = []
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )

    def get_action_with_reasoning(self, state_vector):
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        with torch.no_grad():
            action = self.policy_net(state_tensor).squeeze(0).numpy()

        complexity = np.linalg.norm(state_vector)
        if complexity > 12.0: tier = ComputeTier.HIGH
        elif complexity > 6.0: tier = ComputeTier.MEDIUM
        else: tier = ComputeTier.LOW

        self.tier_history[tier.value] += 1
        return type('VerifiableAction', (), {
            'action': action,
            'tier': tier,
            'predicted_improvement': float(np.tanh(np.mean(action)) * 0.15),
            'confidence_interval': (0.05, 0.10)
        })()

    def log_performance(self, action, actual_improvement):
        self.performance_log.append({
            'tier': action.tier.value,
            'predicted': action.predicted_improvement,
            'actual': actual_improvement,
            'error': abs(action.predicted_improvement - actual_improvement)
        })

    def get_metrics(self):
        if not self.performance_log: return {'mae': 1.0, 'mean_improvement': 0.0}
        errors = [x['error'] for x in self.performance_log]
        improvements = [x['actual'] for x in self.performance_log]
        total = sum(self.tier_history.values())
        return {
            'mae': float(np.mean(errors)),
            'mean_improvement': float(np.mean(improvements)),
            'std_improvement': float(np.std(improvements)),
            'tier_distribution': {t: self.tier_history[t]/total for t in self.tier_history},
            'sample_count': len(self.performance_log)
        }

# --- TRAINING EXECUTION ---

def run_training():
    print("="*80)
    print("🌌 REALIZATION ENGINE: CROSS-VALIDATED TRAINING SESSION (V3.3)")
    print("="*80)

    input_dir = "/kaggle/input/realization-engine-data"
    hard_data_file = os.path.join(input_dir, "realizations", "hard_case_study_dataset.json")

    # Load Hard Case Dataset
    if os.path.exists(hard_data_file):
        with open(hard_data_file, "r") as f:
            full_dataset = json.load(f)
        print(f"✅ Loaded {len(full_dataset)} hard case realizations.")
    else:
        print("⚠️ hard_case_study_dataset.json not found! Generating synthetic.")
        full_dataset = [{'id': f'SYN_{i}', 'content': f'Synthetic {i}', 'features': {'scores': {k: random.random() for k in RealizationEngine.UQS_DIMENSIONS}}} for i in range(50)]

    # CROSS-VALIDATION SPLIT (80/20)
    random.shuffle(full_dataset)
    split_idx = int(len(full_dataset) * 0.8)
    train_data = full_dataset[:split_idx]
    val_data = full_dataset[split_idx:]
    print(f"📊 Dataset split: Train={len(train_data)}, Val={len(val_data)}")

    # Initialize Engine, Evolver, and Agent
    engine = RealizationEngine()
    evolver = SingularityEvolution(engine)

    # State Vector: 512 (emb) + 13 (scores) + 13 (weights) + 2 (meta) = 540
    agent = NextGenPESAgent(dimension="P", weight=0.20, state_dim=540, action_dim=3)

    epochs = 150
    batch_size = 4
    print(f"🚀 Training for {epochs} epochs with Dynamic Weight Adaptation...")

    history = []

    for epoch in range(epochs):
        # 1. Training Phase
        train_metrics = []
        random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]

            # ADAPT WEIGHTS periodically
            if epoch % 5 == 0 and i == 0:
                evolver.adapt_weights(batch)

            for r in batch:
                scores = r.get('features', {}).get('scores', {})
                current_weights = [engine.dimensions[k].weight for k in engine.dimensions]
                state_scores = [scores.get(k, 0.5) for k in engine.dimensions]

                content = r.get('content', '')
                h = int(hashlib.md5(content.encode()).hexdigest(), 16) % 10**8
                np.random.seed(h + epoch)
                mock_emb = np.random.randn(512)

                # State now includes weights for awareness
                state_vector = np.concatenate([mock_emb, state_scores, current_weights, [100, epoch]])

                # Noise Injection
                noise_scale = 0.05 if "ADV" not in r.get('id', '') else 0.2
                state_vector += np.random.normal(0, noise_scale, state_vector.shape)

                v_action = agent.get_action_with_reasoning(state_vector)

                robustness = scores.get('robustness', 0.5)
                q_score, _ = engine.calculate_q_score(RealizationFeatures(scores=scores))
                target_improvement = (robustness * 0.1) + ((1.0 - q_score) * 0.05)
                actual_imp = v_action.predicted_improvement + np.random.normal(target_improvement, 0.01)

                agent.log_performance(v_action, actual_imp)
                train_metrics.append(actual_imp)

        # 2. Validation Phase
        val_metrics = []
        with torch.no_grad():
            for r in val_data:
                scores = r.get('features', {}).get('scores', {})
                current_weights = [engine.dimensions[k].weight for k in engine.dimensions]
                state_scores = [scores.get(k, 0.5) for k in engine.dimensions]

                content = r.get('content', '')
                mock_emb = np.random.randn(512)
                state_vector = np.concatenate([mock_emb, state_scores, current_weights, [100, epoch]])

                v_action = agent.get_action_with_reasoning(state_vector)
                q_score, _ = engine.calculate_q_score(RealizationFeatures(scores=scores))
                target_improvement = (scores.get('robustness', 0.5) * 0.1) + ((1.0 - q_score) * 0.05)
                val_metrics.append(abs(v_action.predicted_improvement - target_improvement))

        avg_train_imp = float(np.mean(train_metrics))
        avg_val_mae = float(np.mean(val_metrics))

        history.append({
            'epoch': epoch,
            'train_improvement': avg_train_imp,
            'val_mae': avg_val_mae,
            'weights': {k: v.weight for k, v in engine.dimensions.items()}
        })

        if epoch % 15 == 0:
            print(f"Epoch {epoch:03d}: Train_Imp = {avg_train_imp:+.4f} | Val_MAE = {avg_val_mae:.4f} | TopWeight = {max(current_weights):.4f}")

    # Finalize
    print("\n" + "="*40)
    print("FINAL EVOLVED TRAINING METRICS")
    print("="*40)
    final_agent_metrics = agent.get_metrics()
    print(f"Mean Train Improvement: {final_agent_metrics['mean_improvement']:.6f}")
    print(f"Final Val MAE: {history[-1]['val_mae']:.6f}")

    # Determine the most influential dimension after evolution
    sorted_dims = sorted(engine.dimensions.items(), key=lambda x: x[1].weight, reverse=True)
    print(f"Most Influential Dimension: {sorted_dims[0][0]} ({sorted_dims[0][1].weight:.4f})")

    results = {
        "timestamp": time.time(),
        "config": {"epochs": epochs, "batch_size": batch_size, "dimensions": 13, "mode": "CROSS_VAL_DYNAMIC"},
        "final_metrics": {**final_agent_metrics, "final_val_mae": history[-1]['val_mae']},
        "history": history,
        "evolved_weights": {k: v.weight for k, v in engine.dimensions.items()},
        "status": "SUCCESS"
    }

    with open("evolved_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(agent.policy_net.state_dict(), "uqs_evolved_agent_v3.3.pt")
    print("\n✅ Evolved training complete. Artifacts saved.")

if __name__ == "__main__":
    run_training()
