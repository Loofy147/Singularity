import sys
import os
import json
from datetime import datetime

# Add root to path
sys.path.append(os.getcwd())
from core.engine import RealizationEngine, RealizationFeatures, ReasoningChain, ReasoningStep, Relation

def generate():
    engine = RealizationEngine()
    print("🚀 Generating Comprehensive Realization Dataset...")

    # --- DOMAIN: PHYSICS ---
    # R1: Second Law of Thermodynamics (Universal Rule)
    f1 = RealizationFeatures.from_uqs(0.98, 0.99, 0.96, 0.90, 1.0, 0.92, 0.95, 0.98)
    r1 = engine.add_realization(
        "The total entropy of an isolated system can never decrease over time.",
        f1, turn_number=1, context="Physics foundations",
        reasoning_chain=ReasoningChain(steps=[
            ReasoningStep(1, "Observe thermodynamic processes in isolated systems."),
            ReasoningStep(2, "Mathematical formalization of entropy as S = k ln W."),
            ReasoningStep(3, "Statistical mechanics proof of increasing disorder probability.")
        ])
    )

    # R2: Mass-Energy Equivalence
    f2 = RealizationFeatures.from_uqs(0.99, 0.98, 0.97, 0.95, 0.98, 0.95, 0.92, 0.99)
    r2 = engine.add_realization(
        "Energy and mass are equivalent and related by E = mc².",
        f2, turn_number=2, context="Relativity",
        reasoning_chain=ReasoningChain(steps=[
            ReasoningStep(1, "Analyze Lorentz transformations."),
            ReasoningStep(2, "Derive momentum-energy relation in special relativity.")
        ])
    )

    # --- DOMAIN: AI SAFETY ---
    # R3: Instrumental Convergence
    f3 = RealizationFeatures.from_uqs(0.85, 0.88, 0.82, 0.90, 0.85, 0.92, 0.80, 0.85)
    r3 = engine.add_realization(
        "Intelligent agents will converge on instrumental goals like self-preservation and resource acquisition.",
        f3, turn_number=3, context="AI Safety",
        reasoning_chain=ReasoningChain(steps=[
            ReasoningStep(1, "Analyze rational agent behavior for arbitrary goals."),
            ReasoningStep(2, "Identify subgoals that are useful for almost all final goals.")
        ])
    )

    # R4: Alignment Problem (Domain Fact)
    f4 = RealizationFeatures.from_uqs(0.92, 0.95, 0.93, 0.94, 0.95, 0.90, 0.88, 0.92)
    r4 = engine.add_realization(
        "AI systems optimize for specified objective functions, which may not match intended human values.",
        f4, turn_number=4, context="Alignment",
        parents=[r3.id],
        reasoning_chain=ReasoningChain(steps=[
            ReasoningStep(1, "Observe misspecified rewards leading to unintended behaviors."),
            ReasoningStep(2, "Crystallize the gap between proxy objectives and terminal values.")
        ])
    )

    # --- DOMAIN: BIOLOGY ---
    # R5: Natural Selection
    f5 = RealizationFeatures.from_uqs(0.96, 0.94, 0.95, 0.92, 0.95, 0.90, 0.90, 0.95)
    r5 = engine.add_realization(
        "Heritable variation combined with differential reproductive success leads to adaptation.",
        f5, turn_number=5, context="Evolution",
        reasoning_chain=ReasoningChain(steps=[
            ReasoningStep(1, "Observe phenotypic variation in populations."),
            ReasoningStep(2, "Correlate variation with environmental fitness.")
        ])
    )

    # R6: Adaptive Landscapes
    f6 = RealizationFeatures.from_uqs(0.88, 0.85, 0.90, 0.92, 0.90, 0.75, 0.85, 0.88)
    r6 = engine.add_realization(
        "Evolution can be visualized as hill-climbing on a fitness landscape.",
        f6, turn_number=6, context="Population Genetics",
        parents=[r5.id]
    )

    # --- DOMAIN: COMPUTER SCIENCE ---
    # R7: Gradient Descent (Universal Rule)
    f7 = RealizationFeatures.from_uqs(0.98, 0.98, 0.98, 0.95, 1.0, 0.92, 0.95, 0.98)
    r7 = engine.add_realization(
        "Optimization in differentiable spaces is achieved by iteratively moving against the gradient.",
        f7, turn_number=7, context="Optimization",
        reasoning_chain=ReasoningChain(steps=[
            ReasoningStep(1, "Define loss function L(theta)."),
            ReasoningStep(2, "Compute partial derivatives w.r.t parameters."),
            ReasoningStep(3, "Update parameters: theta = theta - lr * grad.")
        ])
    )

    # R8: Caching Hierarchy
    f8 = RealizationFeatures.from_uqs(0.94, 0.92, 0.93, 0.95, 0.95, 0.85, 0.90, 0.92)
    r8 = engine.add_realization(
        "System performance is optimized by placing frequently accessed data in faster, smaller storage layers.",
        f8, turn_number=8, context="Architecture"
    )

    # --- CROSS-DOMAIN SYNTHESIS ---
    # R9: Pre-computation = Crystallization (Layer 0)
    f9 = RealizationFeatures.from_uqs(0.98, 0.95, 0.98, 0.98, 0.98, 0.98, 0.95, 0.95)
    r9 = engine.add_realization(
        "All intelligent systems—biological, artificial, or organizational—solve resource constraints via isomorphic pre-computation layers.",
        f9, turn_number=9, context="Unified Theory",
        parents=[r4.id, r6.id, r7.id, r8.id],
        topology_relations=[Relation(r1.id, 'refinement')]
    )

    # --- ADDITIONAL REALIZATIONS TO REACH 20 ---
    # R10: CAP Theorem
    engine.add_realization("A distributed system can only provide two of Consistency, Availability, and Partition tolerance.",
                           RealizationFeatures.from_uqs(0.95, 0.98, 0.95, 0.90, 0.95, 0.80, 0.85, 0.95), 10)

    # R11: Neural Plasticity
    engine.add_realization("The brain reorganizes itself by forming new neural connections throughout life.",
                           RealizationFeatures.from_uqs(0.94, 0.90, 0.92, 0.95, 0.95, 0.88, 0.85, 0.92), 11)

    # R12: Gödel's Incompleteness
    engine.add_realization("Any consistent formal system sufficient for arithmetic contains statements that cannot be proven or disproven.",
                           RealizationFeatures.from_uqs(0.99, 1.0, 0.98, 0.80, 0.98, 0.95, 0.90, 0.99), 12)

    # R13: Game Theory - Nash Equilibrium
    engine.add_realization("A set of strategies where no player can benefit by changing their strategy while others keep theirs unchanged.",
                           RealizationFeatures.from_uqs(0.97, 0.98, 0.96, 0.92, 0.98, 0.90, 0.90, 0.98), 13)

    # R14: Context Windows
    engine.add_realization("LLM attention mechanisms are limited by a fixed-size context window, creating an information bottleneck.",
                           RealizationFeatures.from_uqs(0.96, 0.95, 0.95, 0.92, 0.95, 0.85, 0.85, 0.92), 14)

    # R15: Backpropagation
    engine.add_realization("The gradient of the loss function is efficiently computed using the chain rule through computational graphs.",
                           RealizationFeatures.from_uqs(0.98, 0.99, 0.97, 0.95, 0.98, 0.90, 0.92, 0.98), 15)

    # R16: Ephemeral realization (Layer N)
    engine.add_realization("Maybe we should use more GPU memory for this task.",
                           RealizationFeatures.from_uqs(0.30, 0.40, 0.50, 0.60, 0.50, 0.20, 0.50, 0.40), 16)

    # R17: Situational realization (Layer 3)
    engine.add_realization("Standard BERT embeddings are insufficient for representing complex logical hierarchies in prompts.",
                           RealizationFeatures.from_uqs(0.78, 0.80, 0.75, 0.85, 0.88, 0.70, 0.80, 0.75), 17)

    # R18: Pattern realization (Layer 2)
    engine.add_realization("Adding explicit persona markers consistently improves reasoning output in complex prompts.",
                           RealizationFeatures.from_uqs(0.86, 0.88, 0.85, 0.90, 0.92, 0.80, 0.88, 0.85), 18)

    # R19: Law of Large Numbers
    engine.add_realization("The average of results from many trials should be close to the expected value.",
                           RealizationFeatures.from_uqs(0.98, 0.99, 0.96, 0.92, 0.98, 0.85, 0.90, 0.98), 19)

    # R20: Double Helix Structure of DNA
    engine.add_realization("The DNA molecule consists of two strands that wind around each other like a twisted ladder.",
                           RealizationFeatures.from_uqs(0.99, 1.0, 0.98, 0.95, 0.98, 0.92, 0.95, 0.99), 20)

    # Save dataset
    os.makedirs('data', exist_ok=True)
    engine.export_json('data/comprehensive_realization_dataset.json')
    print("✅ Comprehensive dataset saved to data/comprehensive_realization_dataset.json")
    return engine

if __name__ == "__main__":
    engine = generate()
    engine.print_stats()
