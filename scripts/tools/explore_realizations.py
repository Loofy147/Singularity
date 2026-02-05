import sys
import os
import json

# Add root to path
sys.path.append(os.getcwd())
from core.engine import RealizationEngine

def explore():
    if not os.path.exists('data/realizations/realizations.json'):
        print("Error: data/realizations/realizations.json not found. Run precompute_realizations.py first.")
        return

    # In my version I didn't implement import_json yet, let's add it or just load manually
    with open('data/realizations/realizations.json', 'r') as f:
        data = json.load(f)

    print("\n" + "="*60)
    print("EXPLORING CRYSTALLIZED REALIZATIONS")
    print("="*60)
    print(f"Total: {data['stats']['total_realizations']}")
    print(f"Avg Q: {data['stats']['avg_q_score']:.4f}")

    print("\nTop Realizations:")
    realizations = data['realizations']
    realizations.sort(key=lambda x: x['q_score'], reverse=True)

    for r in realizations[:5]:
        print(f"\n[{r['layer']}] Q={r['q_score']:.4f}")
        print(f"Content: {r['content']}")
        if r['parents']:
            print(f"Parents: {', '.join(r['parents'])}")

if __name__ == "__main__":
    explore()
