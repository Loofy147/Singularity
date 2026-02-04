import sys
import os
import json
import logging

# Add root to path
sys.path.append(os.getcwd())
from core.omega import OMEGAOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_prompts():
    logger.info("🚀 Starting Optimization of Advanced Research Prompts")

    prompts = [
        """### PROMPT 1: COMPREHENSIVE MULTI-DOMAIN DATASET
Generate a comprehensive dataset of knowledge realizations for a 'Realization Crystallization Engine'. The dataset should include 20 realizations across at least 4 domains (e.g., Physics, AI Safety, Philosophy, Law). Each realization must include:
1. Content: A clear statement of the insight.
2. Feature Scores (0-1): grounding, certainty, structure, applicability, coherence, generativity, presentation, temporal.
3. Layer: Based on Q-score thresholds (0: Universal, 1: Domain, 2: Pattern, 3: Situational, N: Ephemeral).
4. Topology: Parent-child relationships and typed relations (derivation, synthesis, contradiction, refinement).
5. Reasoning: A step-by-step chain of thought explaining the realization.
Output the result in a valid JSON format compatible with the RealizationEngine schema.
Requirement: All realizations should aim for Q > 0.85.""",

        """### PROMPT 2: SPECIALIZED MEDICAL REALIZATIONS
Generate a high-quality dataset of 15 medical knowledge realizations focusing on Pharmacology, Immunology, and Neuroscience.
Each realization must reach Q > 0.85 and include:
- Evidence-based grounding.
- Clear structural hierarchy.
- Direct clinical applicability.
- Reasoning chain linking physiological mechanisms to outcomes.
Output in JSON format for the RealizationEngine.""",

        """### PROMPT 3: SPECIALIZED LEGAL & ETHICAL REALIZATIONS
Generate a dataset of 15 realizations covering Jurisprudence, AI Ethics, and International Law.
Focus on:
- Precedent-based grounding.
- Complex topology (contradictions and refinements).
- Layer 0 and Layer 1 status for core legal principles.
- Q-score > 0.88 across all entries.
Output in JSON format.""",

        """### PROMPT 4: ECONOMIC & SYSTEMIC REALIZATIONS
Generate a dataset of 15 realizations about Macroeconomics, Game Theory, and Complex Adaptive Systems.
Each entry must include:
- Mathematical foundation in grounding.
- Generativity (بنات افكار) tracking (how these ideas spawn market trends).
- Q > 0.85 with emphasis on temporal resilience.
Output in JSON format.""",

        """### PROMPT 5: META-OPTIMIZATION REALIZATIONS
Generate a dataset of 10 realizations about the Prompt Engineering Score (PES) itself and the optimization of multi-agent systems.
Focus on:
- Self-referential coherence.
- Theoretical convergence of quality frameworks.
- Q > 0.90 for all entries.
Output in JSON format."""
    ]

    omega = OMEGAOrchestrator()

    # We run the cycle. OMEGAOrchestrator.run_cycle uses MultiAgentCoordinator.optimize_knowledge
    # which adds realizations to its engine index.
    results = omega.run_cycle(prompts, target_q=0.92)

    # Extract the realizations generated from OMEGA cycle
    engine = omega.coordinator.engine
    realizations = [r.to_dict() for r in engine.index.values()]

    output_path = "data/optimized_realizations_v3.1.json"
    with open(output_path, "w") as f:
        json.dump(realizations, f, indent=2)

    logger.info(f"✅ Optimization complete. {len(realizations)} realizations saved to {output_path}")

if __name__ == "__main__":
    optimize_prompts()
