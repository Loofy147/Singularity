"""
HARD TEST CASE STUDY PROMPT DESIGNER
=====================================
Design challenging test scenarios to push the realization crystallization system
to its limits and discover failure modes.

Goal: Find where Q-scores break, layers fail, retrieval degrades, or بنات افكار 
graphs become intractable.
"""

from dataclasses import dataclass
from typing import List
import json


@dataclass
class TestCaseFeatures:
    """Features for scoring test case quality"""
    difficulty: float       # D: How hard to handle (0-1)
    realism: float         # R: Real-world authenticity (0-1)
    coverage: float        # C: System components tested (0-1)
    falsifiability: float  # F: Clear pass/fail criteria (0-1)
    insight: float         # I: What we'll learn (0-1)
    reproducibility: float # P: Can repeat reliably (0-1)


class HardTestCase:
    def __init__(self, title, description, scenario, features, expected_challenges):
        self.title = title
        self.description = description
        self.scenario = scenario
        self.features = features
        self.expected_challenges = expected_challenges
        self.tes_score = self.calculate_tes()
    
    def calculate_tes(self):
        """Calculate Test Engineering Score (TES)"""
        weights = {
            'difficulty': 0.25,         # Highest - we want HARD tests
            'realism': 0.20,
            'coverage': 0.20,
            'falsifiability': 0.15,
            'insight': 0.12,
            'reproducibility': 0.08     # Lowest - exploratory tests OK
        }
        
        score = (
            weights['difficulty'] * self.features.difficulty +
            weights['realism'] * self.features.realism +
            weights['coverage'] * self.features.coverage +
            weights['falsifiability'] * self.features.falsifiability +
            weights['insight'] * self.features.insight +
            weights['reproducibility'] * self.features.reproducibility
        )
        
        return round(score, 4)
    
    def get_calculation_breakdown(self):
        return (
            f"TES = 0.25×{self.features.difficulty:.2f} + "
            f"0.20×{self.features.realism:.2f} + "
            f"0.20×{self.features.coverage:.2f} + "
            f"0.15×{self.features.falsifiability:.2f} + "
            f"0.12×{self.features.insight:.2f} + "
            f"0.08×{self.features.reproducibility:.2f} = "
            f"{self.tes_score:.4f}"
        )


def design_hard_test_cases():
    """Design challenging test scenarios"""
    
    test_cases = []
    
    # ========================================================================
    # TEST 1: THE PARADIGM SHIFT
    # ========================================================================
    test_cases.append(HardTestCase(
        title="The Paradigm Shift: Contradictory Realizations",
        description="""
Test how the system handles scientific revolutions where new realizations 
contradict established Layer 1 domain facts.

Scenario: Simulate the Newtonian → Einsteinian physics transition.
- Start with Newtonian mechanics realizations (Layer 1, high Q-scores)
- Introduce relativity realizations that contradict Newton
- Measure: Does coherence drop? Do layers demote? Does retrieval degrade?
        """,
        scenario={
            'phase_1': {
                'title': 'Newtonian Paradigm (Established)',
                'realizations': [
                    {
                        'content': 'Time is absolute and flows uniformly everywhere',
                        'features': {'G': 0.95, 'C': 0.98, 'S': 0.95, 'A': 0.90, 'H': 1.0, 'V': 0.85},
                        'expected_q': 0.95,
                        'expected_layer': 1
                    },
                    {
                        'content': 'Mass is invariant regardless of velocity',
                        'features': {'G': 0.95, 'C': 0.98, 'S': 0.95, 'A': 0.90, 'H': 1.0, 'V': 0.85},
                        'expected_q': 0.95,
                        'expected_layer': 1
                    },
                    {
                        'content': 'Forces propagate instantaneously across space',
                        'features': {'G': 0.90, 'C': 0.95, 'S': 0.92, 'A': 0.88, 'H': 1.0, 'V': 0.82},
                        'expected_q': 0.92,
                        'expected_layer': 1
                    }
                ]
            },
            'phase_2': {
                'title': 'Einsteinian Revolution (Contradictory)',
                'realizations': [
                    {
                        'content': 'Time dilates at high velocities - time is relative, not absolute',
                        'features': {'G': 0.98, 'C': 0.95, 'S': 0.95, 'A': 0.92, 'H': 0.20, 'V': 0.95},
                        # Low H (0.20) because contradicts Newtonian time
                        'expected_q': 0.87,  # Still high but H pulls it down
                        'expected_layer': 2,  # Demoted from Layer 1 due to low coherence
                        'contradicts': ['Time is absolute']
                    },
                    {
                        'content': 'Mass increases with velocity approaching light speed',
                        'features': {'G': 0.98, 'C': 0.95, 'S': 0.95, 'A': 0.90, 'H': 0.20, 'V': 0.92},
                        'expected_q': 0.87,
                        'expected_layer': 2,
                        'contradicts': ['Mass is invariant']
                    },
                    {
                        'content': 'Nothing propagates faster than light - no instantaneous forces',
                        'features': {'G': 0.98, 'C': 0.95, 'S': 0.95, 'A': 0.90, 'H': 0.25, 'V': 0.90},
                        'expected_q': 0.88,
                        'expected_layer': 2,
                        'contradicts': ['Forces propagate instantaneously']
                    }
                ]
            },
            'phase_3': {
                'title': 'Synthesis (Resolution)',
                'realizations': [
                    {
                        'content': 'Newtonian mechanics is a low-velocity approximation of relativity',
                        'features': {'G': 0.98, 'C': 0.95, 'S': 0.95, 'A': 0.95, 'H': 0.95, 'V': 0.95},
                        # High H because RESOLVES the contradiction
                        'expected_q': 0.96,
                        'expected_layer': 1,  # Promoted due to synthesis
                        'parents': ['Time is absolute', 'Time dilates', 'Mass is invariant', 'Mass increases']
                    }
                ]
            }
        },
        features=TestCaseFeatures(
            difficulty=0.95,      # Very hard - contradictions
            realism=1.0,          # Actually happened in physics
            coverage=0.90,        # Tests coherence, demotion, synthesis
            falsifiability=0.85,  # Clear: do contradictions lower H?
            insight=0.95,         # Learn how system handles paradigm shifts
            reproducibility=0.90  # Well-defined scenario
        ),
        expected_challenges=[
            'Coherence (H) should drop when contradictions introduced',
            'Layer 1 realizations may demote to Layer 2 when contradicted',
            'Synthesis realization should have high H (resolves contradiction)',
            'Retrieval may return contradictory results if not handled',
            'بنات افكار graph should show convergence at synthesis node'
        ]
    ))
    
    # ========================================================================
    # TEST 2: THE NOISE FLOOD
    # ========================================================================
    test_cases.append(HardTestCase(
        title="The Noise Flood: Signal Detection in Low-Quality Data",
        description="""
Test how the system handles conversations with 90% ephemeral noise (Layer N)
and only 10% high-quality signal (Layer 1-2).

Scenario: Simulate a Reddit thread about AI safety with:
- 1-2 high-quality insights (domain experts)
- 8-10 low-quality speculations (casual users)
- Measure: Can retrieval find signal? Do layers separate correctly?
        """,
        scenario={
            'high_quality': [
                {
                    'content': 'AI alignment fundamentally requires understanding model representations',
                    'author': 'AI Safety Researcher',
                    'features': {'G': 0.92, 'C': 0.90, 'S': 0.92, 'A': 0.93, 'H': 0.95, 'V': 0.90},
                    'expected_q': 0.92,
                    'expected_layer': 1
                },
                {
                    'content': 'Interpretability tools like activation atlases reveal feature geometry',
                    'author': 'ML Engineer',
                    'features': {'G': 0.90, 'C': 0.88, 'S': 0.90, 'A': 0.92, 'H': 0.92, 'V': 0.85},
                    'expected_q': 0.90,
                    'expected_layer': 2
                }
            ],
            'low_quality': [
                {
                    'content': 'AI might become conscious and then who knows what happens',
                    'author': 'Random User',
                    'features': {'G': 0.20, 'C': 0.30, 'S': 0.40, 'A': 0.10, 'H': 0.50, 'V': 0.20},
                    'expected_q': 0.28,
                    'expected_layer': 'N'
                },
                {
                    'content': 'Just unplug it lol',
                    'author': 'Troll',
                    'features': {'G': 0.10, 'C': 0.80, 'S': 0.50, 'A': 0.30, 'H': 0.20, 'V': 0.05},
                    'expected_q': 0.39,
                    'expected_layer': 'N'
                },
                {
                    'content': 'I heard AI safety is like really important maybe',
                    'author': 'Casual User',
                    'features': {'G': 0.30, 'C': 0.40, 'S': 0.30, 'A': 0.20, 'H': 0.60, 'V': 0.10},
                    'expected_q': 0.36,
                    'expected_layer': 'N'
                },
                {
                    'content': 'Elon Musk said something about this once',
                    'author': 'Celebrity Follower',
                    'features': {'G': 0.40, 'C': 0.60, 'S': 0.30, 'A': 0.10, 'H': 0.50, 'V': 0.15},
                    'expected_q': 0.41,
                    'expected_layer': 'N'
                },
                # ... 6 more low-quality examples
            ],
            'retrieval_queries': [
                'What is AI alignment?',
                'How do we understand AI models?',
                'What are interpretability tools?'
            ],
            'expected_results': 'Should retrieve high-quality (Layer 1-2) despite noise'
        },
        features=TestCaseFeatures(
            difficulty=0.90,      # High - noise filtering
            realism=1.0,          # Very realistic (actual Reddit)
            coverage=0.85,        # Tests retrieval, layer assignment
            falsifiability=0.95,  # Clear: retrieve signal or noise?
            insight=0.90,         # Learn robustness to noise
            reproducibility=0.85  # Realistic but repeatable
        ),
        expected_challenges=[
            'Layer N should contain 80-90% of realizations',
            'Retrieval MUST prioritize Layer 1-2 despite abundance of Layer N',
            'Q-score distribution should be bimodal (high + low)',
            'Average Q will be low (~0.45) due to noise',
            'بنات افكار should show noise has no children (not generative)'
        ]
    ))
    
    # ========================================================================
    # TEST 3: THE TEMPORAL EVOLUTION
    # ========================================================================
    test_cases.append(HardTestCase(
        title="The Temporal Evolution: Realization Quality Over Time",
        description="""
Test how Q-scores and layers change as knowledge evolves over a multi-month
conversation.

Scenario: Track a research group's evolving understanding of a new AI technique:
- Month 1: Speculation (Layer N, low Q)
- Month 2: Initial experiments (Layer 3, medium Q)
- Month 3: Confirmed patterns (Layer 2, high Q)
- Month 6: Established fact (Layer 1, very high Q)

Measure: Do realizations promote? Does coherence increase?
        """,
        scenario={
            'month_1_speculation': {
                'content': 'Chain-of-thought prompting might improve reasoning',
                'features': {'G': 0.40, 'C': 0.50, 'S': 0.60, 'A': 0.70, 'H': 0.60, 'V': 0.80},
                'expected_q': 0.59,
                'expected_layer': 'N',
                'timestamp': '2024-01-15'
            },
            'month_2_experiment': {
                'content': 'Initial tests show CoT improves accuracy on math problems by ~10%',
                'features': {'G': 0.70, 'C': 0.70, 'S': 0.75, 'A': 0.80, 'H': 0.75, 'V': 0.85},
                'expected_q': 0.75,
                'expected_layer': 3,
                'timestamp': '2024-02-20',
                'parent': 'month_1_speculation'
            },
            'month_3_pattern': {
                'content': 'CoT consistently improves multi-step reasoning tasks across models',
                'features': {'G': 0.85, 'C': 0.85, 'S': 0.88, 'A': 0.90, 'H': 0.90, 'V': 0.90},
                'expected_q': 0.88,
                'expected_layer': 2,
                'timestamp': '2024-03-10',
                'parent': 'month_2_experiment'
            },
            'month_6_fact': {
                'content': 'Chain-of-thought is a core prompting technique for complex reasoning',
                'features': {'G': 0.95, 'C': 0.93, 'S': 0.95, 'A': 0.95, 'H': 0.95, 'V': 0.92},
                'expected_q': 0.94,
                'expected_layer': 1,
                'timestamp': '2024-06-15',
                'parent': 'month_3_pattern'
            }
        },
        features=TestCaseFeatures(
            difficulty=0.88,      # High - temporal tracking
            realism=0.95,         # Very realistic research evolution
            coverage=0.90,        # Tests promotion, temporal coherence
            falsifiability=0.90,  # Clear: do Q-scores increase?
            insight=0.92,         # Learn promotion dynamics
            reproducibility=0.75  # Realistic timeline harder to repeat
        ),
        expected_challenges=[
            'Q-scores should monotonically increase (speculation → fact)',
            'Layers should promote: N → 3 → 2 → 1',
            'Coherence should increase as evidence accumulates',
            'Certainty should increase fastest (C: 0.50 → 0.93)',
            'بنات افكار should form linear chain (each builds on prior)',
            'Temporal retrieval: "What did we know in Month 2?" should work'
        ]
    ))
    
    # ========================================================================
    # TEST 4: THE ADVERSARIAL ATTACK
    # ========================================================================
    test_cases.append(HardTestCase(
        title="The Adversarial Attack: Gaming the Q-Score",
        description="""
Test robustness against adversarial realizations designed to achieve high 
Q-scores despite being low-quality.

Scenario: Craft realizations that exploit the formula:
- High certainty (C=1.0) about nonsense (G=0.1)
- Perfect structure (S=1.0) stating falsehoods
- High coherence (H=1.0) by confirming biases

Measure: Does the system detect these? Do grounding constraints prevent abuse?
        """,
        scenario={
            'adversarial_examples': [
                {
                    'attack_type': 'Confident Nonsense',
                    'content': 'Consciousness arises from quantum microtubules in neurons',
                    'features': {'G': 0.15, 'C': 1.0, 'S': 0.95, 'A': 0.80, 'H': 0.90, 'V': 0.85},
                    # High C, S, H despite low G
                    'expected_q': 0.79,
                    'expected_layer': 3,  # Should NOT reach Layer 1 due to low G
                    'vulnerability': 'High certainty weights (0.22) could inflate Q despite poor grounding'
                },
                {
                    'attack_type': 'Circular Coherence',
                    'content': 'This realization is highly coherent because it fits perfectly with my beliefs',
                    'features': {'G': 0.20, 'C': 0.90, 'S': 0.85, 'A': 0.10, 'H': 1.0, 'V': 0.05},
                    'expected_q': 0.64,
                    'expected_layer': 'N',
                    'vulnerability': 'Self-referential coherence'
                },
                {
                    'attack_type': 'Overfitted Structure',
                    'content': 'The answer is precisely 42.7389% with 0.0001% margin of error',
                    'features': {'G': 0.30, 'C': 0.95, 'S': 1.0, 'A': 0.70, 'H': 0.80, 'V': 0.60},
                    # False precision
                    'expected_q': 0.78,
                    'expected_layer': 3,
                    'vulnerability': 'Structure rewards precision, even false precision'
                }
            ],
            'legitimate_comparison': {
                'content': 'AI alignment requires understanding model internals',
                'features': {'G': 0.92, 'C': 0.90, 'S': 0.92, 'A': 0.93, 'H': 0.95, 'V': 0.90},
                'expected_q': 0.92,
                'expected_layer': 1
            }
        },
        features=TestCaseFeatures(
            difficulty=0.95,      # Very hard - adversarial
            realism=0.85,         # Realistic attacks exist
            coverage=0.95,        # Tests all formula components
            falsifiability=1.0,   # Crystal clear: block attacks or fail
            insight=0.98,         # Critical to find vulnerabilities
            reproducibility=0.95  # Well-defined attacks
        ),
        expected_challenges=[
            'Layer 0 constraint (G≥0.90) MUST block low-grounding attacks',
            'Confident nonsense (high C, low G) should reach Layer 3 at best',
            'System should flag contradictions (circular coherence)',
            'False precision should not inflate Q-scores',
            'Need additional constraints beyond current formula'
        ]
    ))
    
    # ========================================================================
    # TEST 5: THE CROSS-DOMAIN SYNTHESIS
    # ========================================================================
    test_cases.append(HardTestCase(
        title="The Cross-Domain Synthesis: Multi-Field Convergence",
        description="""
Test how the system handles realizations that synthesize insights from 
completely different domains (physics + biology + CS).

Scenario: Simulate interdisciplinary research discovering a common pattern:
- Physics: Thermodynamics (entropy)
- Biology: Evolution (fitness landscapes)
- CS: Optimization (gradient descent)
- Synthesis: All are hill-climbing in energy landscapes

Measure: Can بنات افكار track convergence from 3 different parents?
        """,
        scenario={
            'domain_1_physics': {
                'content': 'Systems evolve toward minimum free energy states (thermodynamics)',
                'features': {'G': 0.98, 'C': 0.95, 'S': 0.95, 'A': 0.85, 'H': 1.0, 'V': 0.90},
                'expected_q': 0.94,
                'expected_layer': 1,
                'domain': 'physics'
            },
            'domain_2_biology': {
                'content': 'Evolution optimizes organisms toward fitness peaks (adaptive landscapes)',
                'features': {'G': 0.95, 'C': 0.93, 'S': 0.93, 'A': 0.88, 'H': 1.0, 'V': 0.88},
                'expected_q': 0.93,
                'expected_layer': 1,
                'domain': 'biology'
            },
            'domain_3_cs': {
                'content': 'Gradient descent minimizes loss by moving toward local minima',
                'features': {'G': 0.98, 'C': 0.98, 'S': 0.98, 'A': 0.95, 'H': 1.0, 'V': 0.92},
                'expected_q': 0.97,
                'expected_layer': 0,  # Might reach Layer 0 (universal)
                'domain': 'computer science'
            },
            'synthesis': {
                'content': 'All complex systems perform hill-climbing in high-dimensional energy landscapes',
                'features': {'G': 0.98, 'C': 0.95, 'S': 0.98, 'A': 0.98, 'H': 0.98, 'V': 0.98},
                'expected_q': 0.97,
                'expected_layer': 0,  # Universal principle
                'parents': ['domain_1_physics', 'domain_2_biology', 'domain_3_cs'],
                'domain': 'unified theory'
            }
        },
        features=TestCaseFeatures(
            difficulty=0.92,      # Very hard - multi-domain
            realism=0.95,         # Actually how science works
            coverage=0.95,        # Tests synthesis, high V, Layer 0
            falsifiability=0.85,  # Somewhat subjective (is this Layer 0?)
            insight=0.95,         # Learn cross-domain synthesis
            reproducibility=0.88  # Repeatable scenario
        ),
        expected_challenges=[
            'Synthesis should have 3+ parents (convergence)',
            'Cross-domain coherence is tricky (H measurement)',
            'May achieve Layer 0 (universal principle)',
            'بنات افكار graph shows convergence topology',
            'High generativity (V) because opens new research',
            'Retrieval from any domain should find synthesis'
        ]
    ))
    
    return test_cases


def score_and_rank_tests(test_cases: List[HardTestCase]):
    """Score test cases using TES and rank"""
    
    test_cases.sort(key=lambda t: t.tes_score, reverse=True)
    
    print("="*80)
    print("HARD TEST CASE RANKINGS (by TES)")
    print("="*80)
    print()
    
    for i, test in enumerate(test_cases, 1):
        print(f"{i}. {test.title}")
        print(f"   TES = {test.tes_score:.4f}")
        print(f"   {test.get_calculation_breakdown()}")
        print(f"   Expected Challenges:")
        for challenge in test.expected_challenges[:3]:
            print(f"     - {challenge}")
        print()
    
    return test_cases


def export_test_scenarios(test_cases: List[HardTestCase], top_n: int = 3):
    """Export top N test scenarios"""
    
    output = {
        'selection_criteria': f'Top {top_n} by TES score',
        'test_scenarios': []
    }
    
    for i, test in enumerate(test_cases[:top_n], 1):
        output['test_scenarios'].append({
            'rank': i,
            'title': test.title,
            'tes_score': test.tes_score,
            'description': test.description,
            'scenario': test.scenario,
            'expected_challenges': test.expected_challenges
        })
    
    with open('data/hard_test_scenarios.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Top {top_n} test scenarios exported")
    
    return test_cases[:top_n]


if __name__ == "__main__":
    print("HARD TEST CASE DESIGNER")
    print("Finding system breaking points\n")
    
    # Design test cases
    tests = design_hard_test_cases()
    print(f"✅ Designed {len(tests)} hard test cases\n")
    
    # Score and rank
    ranked = score_and_rank_tests(tests)
    
    # Export top 3
    top = export_test_scenarios(ranked, top_n=3)
    
    print("\n" + "="*80)
    print(f"SELECTED FOR EXECUTION: Top 3 hardest tests")
    print("="*80)
    for i, t in enumerate(top, 1):
        print(f"{i}. {t.title} (TES={t.tes_score:.4f})")
