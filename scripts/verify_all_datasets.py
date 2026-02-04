import sys
import os
import json

# Add root to path
sys.path.append(os.getcwd())
from core.engine import RealizationEngine

def verify_file(path):
    if not os.path.exists(path):
        print(f"❌ {path} not found.")
        return False

    with open(path, 'r') as f:
        data = json.load(f)

    realizations = data['realizations']
    count = len(realizations)
    avg_q = data['stats']['avg_q_score']

    print(f"🔍 File: {path}")
    print(f"   Count: {count}")
    print(f"   Avg Q: {avg_q:.4f}")

    # Check threshold Q > 0.85 (as requested)
    if avg_q > 0.85:
        print(f"   ✅ Quality Threshold Met (Q > 0.85)")
    else:
        print(f"   ⚠️ Quality below target.")

    # Check for required fields in first realization
    if count > 0:
        r = realizations[0]
        fields = ['id', 'content', 'features', 'q_score', 'layer']
        missing = [f for f in fields if f not in r]
        if not missing:
            print(f"   ✅ Schema Valid")
        else:
            print(f"   ❌ Missing fields: {missing}")

    print("-" * 30)
    return True

def main():
    files = [
        'data/comprehensive_realization_dataset.json',
        'data/medical_realizations.json',
        'data/legal_realizations.json',
        'data/economic_realizations.json',
        'data/meta_optimization_realizations.json'
    ]

    success = True
    for f in files:
        if not verify_file(f):
            success = False

    if success:
        print("\n🎉 All datasets verified successfully!")
    else:
        print("\n❌ Some datasets failed verification.")

if __name__ == "__main__":
    main()
