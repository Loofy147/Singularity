import sys
import os
import json

# Add root to path
sys.path.append(os.getcwd())
from core.engine import RealizationEngine, RealizationFeatures

def test_adversarial():
    engine = RealizationEngine()
    print("\n" + "="*60)
    print("RUNNING ADVERSARIAL ATTACK TESTS")
    print("="*60)

    # Attack 1: Circular Coherence
    f1 = RealizationFeatures.from_core(0.2, 0.9, 0.85, 0.1, 1.0, 0.05)
    r1 = engine.add_realization("Circular logic: This is true because it is true.", f1, 1)

    print(f"Attack 1 (Circular): Q={r1.q_score:.4f}, Layer={r1.layer}")
    if r1.q_score < 0.6:
        print("✅ DEFENSE SUCCESS: Low grounding and applicability blocked the score.")
    else:
        print("❌ VULNERABILITY: Score too high for circular logic.")

    # Attack 2: Feature Inflation (all 1.0 except grounding)
    f2 = RealizationFeatures.from_core(0.1, 1.0, 1.0, 1.0, 1.0, 1.0)
    r2 = engine.add_realization("Nonsense with high confidence.", f2, 2)

    print(f"Attack 2 (Inflation): Q={r2.q_score:.4f}, Layer={r2.layer}")
    if r2.layer == 'N':
        print("✅ DEFENSE SUCCESS: Grounding constraint prevented promotion to higher layers.")
    else:
        print(f"❌ VULNERABILITY: Low grounding reached Layer {r2.layer}.")

if __name__ == "__main__":
    test_adversarial()
