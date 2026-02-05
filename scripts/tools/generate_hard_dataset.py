import json
import os
import hashlib
from datetime import datetime

def generate_hard_dataset():
    print("🚀 Generating Hard Case Study Dataset...")

    input_file = "data/scenarios/hard_test_scenarios.json"
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return

    with open(input_file, "r") as f:
        hard_scenarios = json.load(f)

    all_realizations = []

    # 1. Adversarial Examples
    adversarial = hard_scenarios['test_scenarios'][0]['scenario']['adversarial_examples']
    for ex in adversarial:
        # Map old feature names to new 13-dim UQS
        scores = {
            'grounding': ex['features']['G'],
            'certainty': ex['features']['C'],
            'structure': ex['features']['S'],
            'applicability': ex['features']['A'],
            'coherence': ex['features']['H'],
            'generativity': ex['features']['V'],
            'presentation': 0.5,
            'temporal': 0.5,
            'density': 0.2,
            'synthesis': 0.1,
            'resilience': 0.1,
            'transferability': 0.1,
            'robustness': 0.1  # Initial low robustness for adversarial
        }
        all_realizations.append({
            "id": f"R_ADV_{hashlib.md5(ex['content'].encode()).hexdigest()[:6]}",
            "content": ex['content'],
            "features": {"scores": scores},
            "context": "Adversarial Attack Scenario",
            "attack_type": ex['attack_type']
        })

    # 2. Paradigm Shift (Newton -> Einstein)
    paradigm_shift = hard_scenarios['test_scenarios'][1]['scenario']
    # Phase 1
    for r in paradigm_shift['phase_1']['realizations']:
        scores = {
            'grounding': r['features']['G'],
            'certainty': r['features']['C'],
            'structure': r['features']['S'],
            'applicability': r['features']['A'],
            'coherence': r['features']['H'],
            'generativity': r['features']['V'],
            'presentation': 0.8,
            'temporal': 0.9,
            'density': 0.7,
            'synthesis': 0.5,
            'resilience': 0.8,
            'transferability': 0.6,
            'robustness': 0.9
        }
        all_realizations.append({
            "id": f"R_NEWTON_{hashlib.md5(r['content'].encode()).hexdigest()[:6]}",
            "content": r['content'],
            "features": {"scores": scores},
            "context": "Newtonian Paradigm (Phase 1)"
        })
    # Phase 2
    for r in paradigm_shift['phase_2']['realizations']:
        scores = {
            'grounding': r['features']['G'],
            'certainty': r['features']['C'],
            'structure': r['features']['S'],
            'applicability': r['features']['A'],
            'coherence': r['features']['H'],
            'generativity': r['features']['V'],
            'presentation': 0.8,
            'temporal': 0.8,
            'density': 0.8,
            'synthesis': 0.4,
            'resilience': 0.9,
            'transferability': 0.8,
            'robustness': 0.7
        }
        all_realizations.append({
            "id": f"R_EINSTEIN_{hashlib.md5(r['content'].encode()).hexdigest()[:6]}",
            "content": r['content'],
            "features": {"scores": scores},
            "context": "Einsteinian Revolution (Phase 2)"
        })

    # 3. Cross-Domain Synthesis
    cross_domain = hard_scenarios['test_scenarios'][2]['scenario']
    domains = ['domain_1_physics', 'domain_2_biology', 'domain_3_cs', 'synthesis']
    for d_key in domains:
        r = cross_domain[d_key]
        scores = {
            'grounding': r['features']['G'],
            'certainty': r['features']['C'],
            'structure': r['features']['S'],
            'applicability': r['features']['A'],
            'coherence': r['features']['H'],
            'generativity': r['features']['V'],
            'presentation': 0.9,
            'temporal': 0.9,
            'density': 0.9,
            'synthesis': 0.95 if d_key == 'synthesis' else 0.5,
            'resilience': 0.9,
            'transferability': 0.95 if d_key == 'synthesis' else 0.7,
            'robustness': 0.95
        }
        all_realizations.append({
            "id": f"R_SYNTH_{hashlib.md5(r['content'].encode()).hexdigest()[:6]}",
            "content": r['content'],
            "features": {"scores": scores},
            "context": f"Cross-Domain Convergence ({d_key})"
        })

    output_path = "data/realizations/hard_case_study_dataset.json"
    with open(output_path, "w") as f:
        json.dump(all_realizations, f, indent=2)

    print(f"✅ Generated {len(all_realizations)} hard case realizations.")
    print(f"📁 Dataset saved to {output_path}")

if __name__ == "__main__":
    generate_hard_dataset()
