import sys
import os
import json

# Add root to path
sys.path.append(os.getcwd())
from core.engine import RealizationEngine, RealizationFeatures

def crystallize():
    print("💎 Crystallizing Theoretical Skills from Synthesis Document...")
    engine = RealizationEngine()

    skills = [
        {
            "id": "SKILL_transfer_learning",
            "content": "Cross-domain knowledge application: Ability to transfer learned patterns from one domain (e.g., physics) to another (e.g., economics) using shared mathematical structures.",
            "q_target": 0.946,
            "layer": 0,
            "grounding": 0.98
        },
        {
            "id": "SKILL_universal_problem_solving",
            "content": "Domain-agnostic problem structuring: Hierarchical decomposition and recursive refinement applied to novel challenges across any functional domain.",
            "q_target": 0.946,
            "layer": 0,
            "grounding": 0.98
        },
        {
            "id": "SKILL_interactive_visual_design",
            "content": "Iterative UI/UX and graphic generation: Creating modular, scalable interface designs using AI-powered synthesis and user-feedback loops.",
            "q_target": 0.900,
            "layer": 1,
            "grounding": 0.85
        },
        {
            "id": "SKILL_metacognitive_awareness",
            "content": "Real-time reasoning self-supervision: Monitoring, evaluating, and regulating thinking processes concurrently with the reasoning chain to prevent bias and errors.",
            "q_target": 0.890,
            "layer": 2,
            "grounding": 0.80
        },
        {
            "id": "SKILL_temporal_coherence",
            "content": "Context maintenance over 100+ turns: Resolving explicit and implicit references to the past while maintaining consistency through context snapshots.",
            "q_target": 0.870,
            "layer": 2,
            "grounding": 0.75
        }
    ]

    for s in skills:
        # Create features to match target Q approximately
        val = s["q_target"]
        feat_scores = {
            "grounding": s["grounding"],
            "certainty": val,
            "structure": val,
            "applicability": val,
            "coherence": val,
            "generativity": val,
            "presentation": val,
            "temporal": val,
            "density": val,
            "synthesis": val,
            "resilience": val,
            "transferability": val,
            "robustness": val
        }

        features = RealizationFeatures(scores=feat_scores)
        r = engine.add_realization(
            content=s["content"],
            features=features,
            turn_number=100,
            context="Theoretical Synthesis Document"
        )
        # Override ID to match skill identifier if possible, but engine generates its own.
        # We'll just print it.
        print(f"✅ Crystallized {s['id']} -> {r.id} (Q={r.q_score:.3f}, Layer={r.layer})")

    # Export to the main realizations file
    engine.export_json("data/realizations/realizations.json")
    print(f"\n✅ All theoretical skills exported to data/realizations/realizations.json")

if __name__ == "__main__":
    crystallize()
