import sys
import os
import json

# Add root to path
sys.path.append(os.getcwd())
from core.engine import RealizationEngine

def verify():
    path = 'data/comprehensive_realization_dataset.json'
    if not os.path.exists(path):
        print(f"❌ Error: {path} not found.")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    print(f"🔍 Verifying dataset: {path}")

    # 1. Check counts
    realizations = data['realizations']
    count = len(realizations)
    if count == 20:
        print(f"✅ Count matches: {count}/20")
    else:
        print(f"⚠️ Count mismatch: {count}/20")

    # 2. Check fields
    required_fields = ['id', 'content', 'features', 'q_score', 'layer', 'reasoning_chain']
    missing_fields = []
    for r in realizations:
        for field in required_fields:
            if field not in r:
                missing_fields.append((r['id'], field))

    if not missing_fields:
        print("✅ All realizations have required fields.")
    else:
        print(f"❌ Missing fields: {missing_fields[:5]}")

    # 3. Check topology (Parent-Child)
    synthesis_r = next((r for r in realizations if "intelligent systems" in r['content'].lower()), None)
    if synthesis_r:
        parents = synthesis_r['parents']
        print(f"✅ Synthesis realization {synthesis_r['id']} has {len(parents)} parents.")
        if len(parents) >= 3:
            print("✅ Cross-domain convergence verified.")
        else:
            print("⚠️ Synthesis has fewer than 3 parents.")
    else:
        print("⚠️ Synthesis realization not found.")

    # 4. Check layer distribution
    dist = data['stats']['layer_distribution']
    print(f"📊 Layer distribution: {dist}")
    if dist['0'] > 0 and dist['N'] > 0:
        print("✅ Dataset covers full spectrum (Layer 0 to N).")
    else:
        print("⚠️ Dataset missing layers.")

    print("\n✅ Verification complete!")

if __name__ == "__main__":
    verify()
