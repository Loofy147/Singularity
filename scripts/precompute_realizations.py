import sys
import os
import json

# Add root to path
sys.path.append(os.getcwd())
from core.engine import RealizationEngine, RealizationFeatures

def precompute():
    engine = RealizationEngine()

    # Core realizations from the original conversation
    # r1: Context windows
    f1 = RealizationFeatures.from_core(0.96, 0.94, 0.95, 0.92, 0.95, 0.90)
    r1 = engine.add_realization(
        "Context windows are finite and information can be lost",
        f1, turn_number=1
    )

    # r2: Crystallization process
    f2 = RealizationFeatures.from_core(0.95, 0.93, 0.94, 0.91, 0.94, 0.92)
    r2 = engine.add_realization(
        "Realizations crystallize into layers: Rules -> Facts -> Patterns -> Insights",
        f2, turn_number=5, parents=[r1.id]
    )

    # r3: Q-Score
    f3 = RealizationFeatures.from_core(0.98, 0.90, 0.95, 0.95, 0.97, 0.88)
    r3 = engine.add_realization(
        "Realization quality can be scored: Q = weighted sum of features",
        f3, turn_number=10, parents=[r2.id]
    )

    # r4: Computability
    f4 = RealizationFeatures.from_core(0.96, 0.92, 0.94, 0.93, 0.96, 0.90)
    r4 = engine.add_realization(
        "Realizations are computable and can be treated as parameters",
        f4, turn_number=15, parents=[r3.id]
    )

    # Save to data/realizations.json
    os.makedirs('data', exist_ok=True)
    engine.export_json('data/realizations.json')
    print("✅ Crystallized knowledge saved to data/realizations.json")
    return engine

if __name__ == "__main__":
    engine = precompute()
    engine.print_stats()
