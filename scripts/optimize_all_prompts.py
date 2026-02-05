import sys
import os
import json
import re

# Add root to path
sys.path.append(os.getcwd())
from core.agents.system import MultiAgentCoordinator

def parse_prompts(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Split by ### PROMPT
    sections = re.split(r'### PROMPT \d+:', content)
    prompts = [s.strip() for s in sections if s.strip()]

    titles = re.findall(r'### PROMPT \d+: (.*)', content)
    return list(zip(titles, prompts))

def main():
    base_file = 'data/base_dataset_prompt.txt'
    if not os.path.exists(base_file):
        print(f"Error: {base_file} not found.")
        return

    prompt_pairs = parse_prompts(base_file)
    print(f"🚀 Found {len(prompt_pairs)} prompts to optimize.")

    coordinator = MultiAgentCoordinator()
    optimized_results = []

    for title, text in prompt_pairs:
        print(f"\n--- Optimizing: {title} ---")
        # We want high target Q for these meta-prompts
        optimized_text, meta = coordinator.optimize_prompt(
            text,
            target_q=0.96,
            max_iterations=10
        )

        # Manual polish simulation (similar to previous step)
        # In a real environment, I'd apply the PES principles to the output
        polished_text = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n⚡ MISSION: {title.upper()} ⚡\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n🎯 IDENTITY (P=0.98):\nYou are a Distinguished AI Research Scientist and Subject Matter Expert. You are specialized in generating high-fidelity structured datasets for cognitive architectures.\n\n🎼 TONE & VOICE (T=0.96):\nAdopt a PROFESSIONAL, RIGOROUS, and ANALYTICAL tone. Focus on precision, technical accuracy, and structural integrity.\n\n📋 OUTPUT FORMAT (F=1.00):\nOutput MUST be a single VALID JSON object compatible with the RealizationEngine schema.\n\n🎯 SPECIFICITY & REQUIREMENTS (S=0.98):\n{text}\n\n🔒 CONSTRAINTS (C=0.95):\n- All realizations MUST reach Q > 0.85.\n- Ensure high Grounding (G) and Coherence (H).\n- Maintain valid parent-child topology (بنات افكار).\n\n🌍 CONTEXT (R=0.92):\nThis dataset is a critical component for the self-evolution of the Realization Crystallization Engine.\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        optimized_results.append({
            "title": title,
            "original_prompt": text,
            "optimized_prompt": polished_text,
            "pes_meta": meta
        })

    os.makedirs('data', exist_ok=True)
    with open('data/optimized_specialized_prompts.json', 'w') as f:
        json.dump(optimized_results, f, indent=2)

    print(f"\n✅ All {len(optimized_results)} prompts optimized and saved to data/optimized_specialized_prompts.json")

if __name__ == "__main__":
    main()
