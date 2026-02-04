"""
RESEARCH PROMPT OPTIMIZER
Using PES (Prompt Engineering Score) based on Q-Score framework

Maps Q-Score features to prompt quality:
- G (Grounding) → Research Foundation
- C (Certainty) → Contribution Clarity  
- S (Structure) → Paper Organization
- A (Applicability) → Practical Impact
- H (Coherence) → Cross-Domain Integration
- V (Generativity) → Future Research Potential

PES = 0.18×F + 0.22×C + 0.20×S + 0.18×I + 0.12×H + 0.10×G
where F=Foundation, C=Clarity, S=Structure, I=Impact, H=Harmony, G=Generativity
"""

from dataclasses import dataclass
from typing import List, Dict
import json


@dataclass
class PromptFeatures:
    """Features for scoring research prompts"""
    foundation: float      # F: How well-grounded in existing work (0-1)
    clarity: float         # C: How clear the contribution (0-1)
    structure: float       # S: How well-organized the output will be (0-1)
    impact: float          # I: Practical/theoretical impact (0-1)
    harmony: float         # H: Integration across domains (0-1)
    generativity: float    # G: Spawns future research (0-1)


class ResearchPrompt:
    def __init__(self, title, prompt, features, context):
        self.title = title
        self.prompt = prompt
        self.features = features
        self.context = context
        self.pes_score = self.calculate_pes()
        
    def calculate_pes(self):
        """Calculate Prompt Engineering Score"""
        weights = {
            'foundation': 0.18,
            'clarity': 0.22,      # Highest - clear contribution is critical
            'structure': 0.20,
            'impact': 0.18,
            'harmony': 0.12,
            'generativity': 0.10
        }
        
        score = (
            weights['foundation'] * self.features.foundation +
            weights['clarity'] * self.features.clarity +
            weights['structure'] * self.features.structure +
            weights['impact'] * self.features.impact +
            weights['harmony'] * self.features.harmony +
            weights['generativity'] * self.features.generativity
        )
        
        return round(score, 4)
    
    def get_calculation_breakdown(self):
        """Show PES calculation"""
        return (
            f"PES = 0.18×{self.features.foundation:.2f} + "
            f"0.22×{self.features.clarity:.2f} + "
            f"0.20×{self.features.structure:.2f} + "
            f"0.18×{self.features.impact:.2f} + "
            f"0.12×{self.features.harmony:.2f} + "
            f"0.10×{self.features.generativity:.2f} = "
            f"{self.pes_score:.4f}"
        )


def design_research_prompts():
    """Design highest-quality research prompts from our work"""
    
    prompts = []
    
    # ========================================================================
    # PROMPT 1: Computational Epistemology
    # ========================================================================
    prompts.append(ResearchPrompt(
        title="Computational Epistemology: Realizations as Computable Knowledge Structures",
        prompt="""
Write a comprehensive academic research paper on "Computational Epistemology: Realizations as Computable Knowledge Structures"

CORE THESIS:
Knowledge acquisition moments (realizations) are not ephemeral cognitive events but computable structures that can be:
1. Quantified via multi-dimensional feature vectors (G,C,S,A,H,V)
2. Scored using weighted quality functions (Q-scores)
3. Organized into hierarchical layers (0→1→2→3→N)
4. Retrieved via graph-based traversal
5. Reproduced via parent-child relationships (بنات افكار)

REQUIRED SECTIONS:
1. Abstract (200 words)
2. Introduction: The Problem of Knowledge Crystallization
3. Theoretical Framework: From Procedural to Declarative Knowledge
4. The Q-Score Formula: Mathematics of Realization Quality
5. Layer Architecture: Hierarchical Knowledge Organization
6. Generativity: بنات افكار (Daughters of Ideas) Graph Theory
7. Implementation: Realization Engine Design
8. Case Study: AI Safety Conversation Analysis (8 realizations, Q=0.8881 avg)
9. Validation: Testing & Performance Metrics
10. Discussion: Implications for AI Epistemology
11. Related Work: Distributed Systems, Caching, Memory Hierarchies
12. Future Work: Automated Extraction, Multi-Agent Crystallization
13. Conclusion
14. References (30+)

EVIDENCE TO INTEGRATE:
- Test case: 8 realizations from AI safety discussion
- Q-scores ranging 0.8246-0.9338
- 100% retrieval accuracy
- 92.9% average coherence
- Layer distribution: 0/1/6/1/0 across layers 0/1/2/3/N
- Graph depth: 7 levels, 11 parent-child relationships
- Alignment problem as highest-Q realization (0.9338)

THEORETICAL GROUNDING:
- Distributed systems: Pre-computation, caching, CDN architectures
- Cognitive science: Declarative vs procedural memory
- Epistemology: Knowledge justification, coherence theory
- Graph theory: DAGs, knowledge graphs, semantic networks

WRITING STYLE:
- Rigorous academic tone
- Mathematical precision (all formulas with derivations)
- Empirical validation (cite test results)
- Cross-disciplinary integration

TARGET VENUE: Nature Computational Science, PNAS, or top-tier AI conference
LENGTH: 8000-10000 words
""",
        features=PromptFeatures(
            foundation=0.95,    # Grounded in systems, cognition, epistemology
            clarity=0.93,       # Very clear thesis: realizations are computable
            structure=0.90,     # Well-defined sections
            impact=0.92,        # High theoretical + practical impact
            harmony=0.95,       # Excellent cross-domain integration
            generativity=0.92   # Opens multiple research directions
        ),
        context="Theoretical foundation paper"
    ))
    
    # ========================================================================
    # PROMPT 2: Pre-Computation = Crystallization
    # ========================================================================
    prompts.append(ResearchPrompt(
        title="Pre-Computation Equals Crystallization: Bridging Distributed Systems and Cognition",
        prompt="""
Write a comprehensive academic research paper on "Pre-Computation Equals Crystallization: A Unified Theory of Knowledge Caching Across Systems and Minds"

CORE THESIS:
Pre-computation in distributed systems and realization crystallization in cognition are mathematically identical processes:
- Both use weighted scoring (efficiency metrics vs Q-scores)
- Both organize into layers (compile/build/deploy/runtime vs 0/1/2/3/N)
- Both have invalidation strategies (TTL/event-based vs coherence decay)
- Both optimize for reuse (cache hit rate vs retrieval frequency)

REQUIRED SECTIONS:
1. Abstract
2. Introduction: Two Worlds, One Pattern
3. Distributed Systems Pre-Computation
   - CDN edge caching
   - Build artifacts (compile, link, package, deploy)
   - Database query results
   - Static site generation
4. Cognitive Crystallization
   - Procedural → declarative knowledge
   - Working memory → long-term memory
   - Insight moments (Aha! experiences)
5. Mathematical Isomorphism
   - Scoring functions (weighted sums)
   - Layer thresholds (quality-based assignment)
   - Invalidation logic (staleness detection)
   - Retrieval optimization (hierarchical search)
6. Unified Framework
   - Generic pre-computation algorithm
   - Applied to both domains
   - Formal proof of equivalence
7. Empirical Validation
   - CDN performance metrics
   - Realization engine test results (Q=0.8881, 100% retrieval)
8. Implications
   - AI systems should use pre-computation for reasoning
   - Human knowledge can be modeled as cached computation
   - Cross-pollination of optimization techniques
9. Related Work
10. Future Work: Hybrid Human-AI Knowledge Systems
11. Conclusion
12. References

EVIDENCE TO INTEGRATE:
- Realization engine: 8 realizations, avg Q=0.8881
- Layer thresholds: 0.95/0.92/0.85/0.75
- Retrieval: O(log n) hierarchical search, 100% accuracy
- Coherence: 92.9% average consistency

THEORETICAL GROUNDING:
- Computer architecture (cache hierarchies)
- Distributed systems (CAP theorem, eventual consistency)
- Cognitive psychology (dual process theory)
- Category theory (functorial mappings)

TARGET VENUE: Science, ACM Computing Surveys, Cognitive Science journal
LENGTH: 7000-9000 words
""",
        features=PromptFeatures(
            foundation=0.98,    # Strong grounding in both systems + cognition
            clarity=0.95,       # Crystal clear: two fields, one math
            structure=0.92,     # Parallel structure across domains
            impact=0.95,        # Huge - unifies two major fields
            harmony=0.98,       # Perfect cross-domain integration
            generativity=0.88   # Opens hybrid systems research
        ),
        context="Cross-domain unification paper"
    ))
    
    # ========================================================================
    # PROMPT 3: Q-Score Measurement Framework
    # ========================================================================
    prompts.append(ResearchPrompt(
        title="The Q-Score Framework: Measuring Realization Quality in AI Systems",
        prompt="""
Write a comprehensive academic research paper on "The Q-Score Framework: A Multi-Dimensional Quality Metric for Knowledge Realizations in AI Systems"

CORE THESIS:
AI systems need a standardized metric for measuring knowledge quality. The Q-score provides:
Q = 0.18×G + 0.22×C + 0.20×S + 0.18×A + 0.12×H + 0.10×V

Where G=Grounding, C=Certainty, S=Structure, A=Applicability, H=Coherence, V=Generativity

REQUIRED SECTIONS:
1. Abstract
2. Introduction: The Knowledge Quality Problem
3. Related Work
   - Precision/recall in ML
   - F1-scores, AUC-ROC
   - Semantic similarity metrics
   - Information quality frameworks
4. The Q-Score Formula
   - Six dimensions (G,C,S,A,H,V)
   - Weight justification (why C=0.22 is highest)
   - Mathematical properties (bounded, additive)
5. Feature Definitions
   - Grounding (0.18): Factual rootedness
   - Certainty (0.22): Self-certifying knowledge ("precision auto")
   - Structure (0.20): Crystallization clarity
   - Applicability (0.18): Actionability
   - Coherence (0.12): Consistency with context
   - Generativity (0.10): بنات افكار potential
6. Layer Thresholds
   - Layer 0: Q≥0.95, G≥0.90 (Universal Rules)
   - Layer 1: Q≥0.92 (Domain Facts)
   - Layer 2: Q≥0.85 (Patterns)
   - Layer 3: Q≥0.75 (Situational)
   - Layer N: Q<0.75 (Ephemeral)
7. Validation Study
   - AI safety conversation (8 realizations)
   - Q-scores: 0.8246-0.9338 (avg 0.8881)
   - Inter-rater reliability (if multiple scorers)
   - Predictive validity (retrieval accuracy: 100%)
8. Comparison to Existing Metrics
   - F1-score: Binary classification only
   - Semantic similarity: No actionability dimension
   - Citation count: Lagging indicator
   - Q-score: Multi-dimensional, real-time
9. Applications
   - RAG systems (rank retrieved knowledge)
   - AI training (filter high-Q data)
   - Knowledge bases (organize by Q-score)
10. Limitations & Future Work
11. Conclusion
12. References

EVIDENCE:
- 8 realizations scored, validated
- Highest: Alignment problem (Q=0.9338)
- Lowest: Sandboxing (Q=0.8246)
- Average coherence: 92.9%
- Retrieval accuracy: 100%

TARGET VENUE: ACM SIGKDD, NeurIPS, ICLR
LENGTH: 6000-8000 words
""",
        features=PromptFeatures(
            foundation=0.92,    # Grounded in ML metrics, information theory
            clarity=0.98,       # Extremely clear: 6 features, formula, done
            structure=0.95,     # Standard metric paper structure
            impact=0.94,        # High - practical metric for AI systems
            harmony=0.85,       # Moderate cross-domain (mostly AI/ML)
            generativity=0.90   # Opens applications in RAG, training, etc.
        ),
        context="Practical metric/framework paper"
    ))
    
    # ========================================================================
    # PROMPT 4: بنات افكار (Generativity)
    # ========================================================================
    prompts.append(ResearchPrompt(
        title="بنات افكار: Graph-Based Knowledge Propagation in Conversational AI",
        prompt="""
Write a comprehensive academic research paper on "بنات افكار (Daughters of Ideas): Graph-Based Modeling of Knowledge Propagation in Multi-Turn Conversations"

CORE THESIS:
Ideas reproduce. Each realization spawns children (بنات افكار) that inherit properties from parents but gain new context. This creates knowledge graphs where:
- Nodes = Realizations (with Q-scores)
- Edges = Parent-child relationships (causality)
- Graph structure = Reasoning chains
- Convergence points = Synthesis moments
- Graph depth = Reasoning complexity

REQUIRED SECTIONS:
1. Abstract
2. Introduction: Ideas as Reproductive Structures
3. Theoretical Framework
   - Memetics (Dawkins)
   - Conceptual blending (Fauconnier & Turner)
   - Graph theory (DAGs, citation networks)
4. The بنات افكار Model
   - Parent-child relationships
   - Property inheritance (coherence constraints)
   - Mutation (context adaptation)
   - Convergence (synthesis)
5. Graph Properties
   - In-degree: How many parents (convergence)
   - Out-degree: How many children (generativity)
   - Depth: Reasoning chain length
   - Branching factor: Idea diversity
6. Case Study: AI Safety Discussion
   - 8 realizations, 11 parent-child relationships
   - R1 (Emergence) → 2 children
   - R7 (Synthesis) ← 4 parents (convergence)
   - Max depth: 7 levels
   - Graph visualization
7. Generativity Analysis
   - Most generative: R1, R2 (2 children each)
   - Highest Q + generativity: R2 (Q=0.9338, V=0.90)
   - Correlation: Q-score vs children count
8. Applications
   - Conversation quality metrics
   - Idea flow visualization
   - Research paper citation analysis
   - Collaborative ideation tools
9. Comparison to Related Models
   - Citation networks (similar structure)
   - Concept maps (similar representation)
   - Discourse graphs (similar analysis)
10. Limitations & Future Work
11. Conclusion
12. References

EVIDENCE:
- 11 parent-child relationships tracked
- R7 synthesis node: 4 parents converged
- Average children per realization: 1.38
- Graph depth: 7 levels

TARGET VENUE: Computational Linguistics, Cognitive Science, CHI
LENGTH: 6000-7000 words
""",
        features=PromptFeatures(
            foundation=0.88,    # Grounded in memetics, graph theory
            clarity=0.90,       # Clear: ideas reproduce via graphs
            structure=0.88,     # Well-organized
            impact=0.85,        # Moderate - more theoretical
            harmony=0.90,       # Good cross-domain (cognition + graphs)
            generativity=0.95   # Very high - opens many research paths
        ),
        context="Generativity-focused paper"
    ))
    
    # ========================================================================
    # PROMPT 5: System Architecture Paper
    # ========================================================================
    prompts.append(ResearchPrompt(
        title="Hierarchical Knowledge Architecture: From Ephemeral to Universal",
        prompt="""
Write a comprehensive academic research paper on "Hierarchical Knowledge Architecture: A Five-Layer System for AI Knowledge Management"

CORE THESIS:
Knowledge should be organized in 5 layers based on quality and stability:
- Layer 0: Universal Rules (Q≥0.95, G≥0.90) - Immutable
- Layer 1: Domain Facts (Q≥0.92) - Rarely change
- Layer 2: Patterns (Q≥0.85) - Context-dependent
- Layer 3: Situational (Q≥0.75) - Temporary
- Layer N: Ephemeral (Q<0.75) - High churn

This mirrors computer architecture (L1/L2/L3 cache, RAM, disk).

REQUIRED SECTIONS:
1. Abstract
2. Introduction: The Knowledge Hierarchy Problem
3. Related Work
   - Memory hierarchies (CPU cache, RAM, disk)
   - Knowledge bases (ontologies, taxonomies)
   - Information architecture
4. The Five-Layer Model
   - Layer definitions
   - Threshold justification
   - Assignment algorithm
5. Layer Properties
   - Access frequency (Layer 0 highest)
   - Mutation rate (Layer N highest)
   - Retrieval priority (hierarchical search)
   - Storage efficiency (Layer 0 most compact)
6. Implementation: Realization Engine
   - Data structures (hash maps per layer)
   - Retrieval algorithm (O(log n))
   - Promotion/demotion logic
7. Case Study Results
   - Layer distribution: 0/1/6/1/0 (AI safety conversation)
   - Quality by layer: L1=0.9338, L2=0.8911, L3=0.8246
   - No Layer 0 (rare, as expected)
8. Performance Analysis
   - Retrieval accuracy: 100%
   - Average Q-score: 0.8881
   - Coherence: 92.9%
9. Applications
   - RAG systems (layer-aware retrieval)
   - Knowledge bases (quality-based organization)
   - AI training data (filter by layer)
10. Comparison to Flat Architectures
11. Future Work: Automated Layer Assignment
12. Conclusion
13. References

EVIDENCE:
- 8 realizations distributed across 3 layers
- Layer 2 dominant (75%) - expected for domain conversations
- Zero ephemeral (all Q≥0.82) - high-quality conversation

TARGET VENUE: VLDB, ACM SIGMOD, IEEE TKDE
LENGTH: 6000-7000 words
""",
        features=PromptFeatures(
            foundation=0.94,    # Strong grounding in systems architecture
            clarity=0.92,       # Clear: 5 layers based on quality
            structure=0.93,     # Very well-organized
            impact=0.90,        # High practical impact for AI systems
            harmony=0.88,       # Good systems + AI integration
            generativity=0.85   # Moderate - mostly architectural
        ),
        context="System architecture paper"
    ))
    
    return prompts


def score_and_rank_prompts(prompts: List[ResearchPrompt]):
    """Score prompts using PES and rank them"""
    
    # Sort by PES score (descending)
    prompts.sort(key=lambda p: p.pes_score, reverse=True)
    
    print("="*80)
    print("RESEARCH PROMPT RANKINGS (by PES)")
    print("="*80)
    print()
    
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt.title}")
        print(f"   PES = {prompt.pes_score:.4f}")
        print(f"   {prompt.get_calculation_breakdown()}")
        print(f"   Context: {prompt.context}")
        print()
    
    return prompts


def export_top_prompts(prompts: List[ResearchPrompt], top_n: int = 3):
    """Export top N prompts for execution"""
    
    top_prompts = prompts[:top_n]
    
    output = {
        'selection_criteria': f'Top {top_n} by PES score',
        'selected_prompts': []
    }
    
    for i, prompt in enumerate(top_prompts, 1):
        output['selected_prompts'].append({
            'rank': i,
            'title': prompt.title,
            'pes_score': prompt.pes_score,
            'features': {
                'foundation': prompt.features.foundation,
                'clarity': prompt.features.clarity,
                'structure': prompt.features.structure,
                'impact': prompt.features.impact,
                'harmony': prompt.features.harmony,
                'generativity': prompt.features.generativity
            },
            'prompt': prompt.prompt
        })
    
    with open('data/top_research_prompts.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Top {top_n} prompts exported to top_research_prompts.json")
    print(f"   Total size: {len(json.dumps(output))} bytes")
    
    return top_prompts


if __name__ == "__main__":
    print("RESEARCH PROMPT OPTIMIZATION SYSTEM")
    print("Using PES (Prompt Engineering Score)")
    print()
    
    # Design prompts
    prompts = design_research_prompts()
    print(f"✅ Designed {len(prompts)} research prompts\n")
    
    # Score and rank
    ranked_prompts = score_and_rank_prompts(prompts)
    
    # Export top 3
    top_prompts = export_top_prompts(ranked_prompts, top_n=3)
    
    print("\n" + "="*80)
    print(f"SELECTED FOR EXECUTION: Top 3 prompts")
    print("="*80)
    for i, p in enumerate(top_prompts, 1):
        print(f"{i}. {p.title} (PES={p.pes_score:.4f})")
