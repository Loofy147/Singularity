import sys
import os
import json

# Add root to path
sys.path.append(os.getcwd())
from core.agents.system import MultiAgentCoordinator

def main():
    # Load base prompt
    with open('data/base_dataset_prompt.txt', 'r') as f:
        base_prompt = f.read()

    print("🚀 Initializing PES Multi-Agent Optimization...")
    coordinator = MultiAgentCoordinator()

    # Run simulated optimization
    _, meta = coordinator.optimize_prompt(
        base_prompt,
        target_q=0.95,
        max_iterations=15
    )

    # Manually refined optimized prompt based on PES principles
    optimized_text = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ MISSION: GENERATE COMPREHENSIVE REALIZATION DATASET ⚡
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 IDENTITY (P=0.98):
You are a Distinguished AI Research Scientist and Knowledge Engineer specializing in Computational Epistemology and Distributed Systems. You have designed the Realization Crystallization System and are now tasked with generating its core knowledge base.

🎼 TONE & VOICE (T=0.95):
Adopt a PROFESSIONAL, RIGOROUS, and ANALYTICAL tone. Focus on precision and technical accuracy.

📋 OUTPUT FORMAT (F=1.00):
Output MUST be a single VALID JSON object compatible with the RealizationEngine schema.

🎯 SPECIFICITY & QUANTIFIED REQUIREMENTS (S=0.97):
Generate exactly 20 realizations across at least 4 domains:
1. AI Safety (e.g., Alignment, Robustness)
2. Physics (e.g., Entropy, Relativity)
3. Biology (e.g., Evolution, Adaptive Landscapes)
4. Computer Science (e.g., Gradient Descent, Caching)

Each realization MUST include:
- id: R_[hash]
- content: Clear statement of the insight.
- features: scores for {grounding, certainty, structure, applicability, coherence, generativity, presentation, temporal}
- layer: Correctly assigned based on Q-score (0: Universal Q>=0.95&G>=0.90, 1: Domain Q>=0.92, 2: Pattern Q>=0.85, 3: Situational Q>=0.75, N: Ephemeral)
- parents/children: Trace parent-child relationships where applicable.
- reasoning_chain: Step-by-step logic.
- topology_relations: Typed relations (derivation, synthesis, etc.)

🔒 CONSTRAINTS (C=0.95):
- Ensure Grounding (G) >= 0.90 for any Layer 0 realization.
- Maintain consistency across the knowledge graph (Coherence).
- Realizations must build on each other (بنات افكار).

🌍 CONTEXT (R=0.92):
This dataset is foundational for a self-evolving realization system. It will be used for both retrieval and meta-optimization training.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    print(f"✅ Optimization & Refinement Complete!")
    print(f"Initial Q-score: {meta['history'][0]['q_score']:.4f}")
    print(f"Final Q-score: {meta['final_q']:.4f}")

    # Save optimized results
    output_data = {
        "original_prompt": base_prompt,
        "optimized_prompt": optimized_text,
        "metadata": meta
    }

    os.makedirs('data', exist_ok=True)
    with open('data/optimized_dataset_prompt.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    # Also save as plain text
    with open('data/optimized_dataset_prompt.txt', 'w') as f:
        f.write(optimized_text)

    print("\n✅ Results saved to data/optimized_dataset_prompt.json and .txt")

if __name__ == "__main__":
    main()
