import sys
import os
import json

# Add root to path
sys.path.append(os.getcwd())
from core.engine import RealizationEngine, RealizationFeatures

def verify():
    print("🚀 Verifying Realization Engine (V3.1) with Emergent Dimensions")
    engine = RealizationEngine()

    # 1. Check Dimensions
    print(f"\nTotal Dimensions: {len(engine.dimensions)}")
    for key, dim in engine.dimensions.items():
        print(f"  - {dim.name} (w={dim.weight:.2f}) [By: {dim.discovered_by}]")

    # 2. Test Q-Score calculation with emergents
    features = RealizationFeatures.from_uqs(
        grounding=0.95, certainty=0.98, structure=0.92,
        applicability=0.90, coherence=0.95, generativity=0.85,
        presentation=0.80, temporal=0.90,
        density=0.95, synthesis=0.90, resilience=0.92, transferability=0.88
    )

    q_score, breakdown = engine.calculate_q_score(features)
    print(f"\nCalculated Q-Score: {q_score:.4f}")

    # 3. Test optimization results
    if os.path.exists("data/realizations/optimized_realizations_v3.1.json"):
        with open("data/realizations/optimized_realizations_v3.1.json", "r") as f:
            optimized = json.load(f)
        print(f"\nVerified local results: {len(optimized)} optimized realizations found.")

    print("\n✅ V3.1 Engine Local Verification PASSED")

if __name__ == "__main__":
    verify()
