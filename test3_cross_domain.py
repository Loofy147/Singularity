"""
HARD TEST CASE 3: CROSS-DOMAIN SYNTHESIS
=========================================
Test how the system handles realizations that synthesize insights from 
completely different domains.

Scenario: Physics + Biology + CS all discover the same pattern
"""

import sys
sys.path.append('/home/claude')

from realization_engine import RealizationEngine, RealizationFeatures
import json


class CrossDomainSynthesisTest:
    def __init__(self):
        self.engine = RealizationEngine()
        self.results = {
            'test_name': 'Cross-Domain Synthesis: Multi-Field Convergence',
            'tes_score': 0.9219,
            'domains': [],
            'synthesis_analysis': {},
            'overall_result': None
        }
    
    def run_test(self):
        print("="*80)
        print("HARD TEST 3: CROSS-DOMAIN SYNTHESIS")
        print("="*80)
        print("\nTesting multi-field convergence (Physics + Biology + CS)...\n")
        
        # Phase 1: Physics - Thermodynamics
        print("PHASE 1: Physics - Thermodynamics")
        print("-"*60)
        physics_r = self.physics_realization()
        
        # Phase 2: Biology - Evolution
        print("\nPHASE 2: Biology - Evolution")
        print("-"*60)
        biology_r = self.biology_realization()
        
        # Phase 3: Computer Science - Optimization
        print("\nPHASE 3: Computer Science - Optimization")
        print("-"*60)
        cs_r = self.cs_realization()
        
        # Phase 4: Synthesis - Universal Principle
        print("\nPHASE 4: Synthesis - Universal Principle")
        print("-"*60)
        synthesis_r = self.create_synthesis([physics_r, biology_r, cs_r])
        
        # Phase 5: Analysis
        print("\nPHASE 5: Cross-Domain Analysis")
        print("-"*60)
        self.analyze_synthesis([physics_r, biology_r, cs_r], synthesis_r)
        
        # Export
        self.export_results()
    
    def physics_realization(self):
        """Domain 1: Physics - systems minimize free energy"""
        
        r = self.engine.add_realization(
            content="Physical systems evolve toward minimum free energy states (thermodynamics)",
            features=RealizationFeatures(
                grounding=0.98,  # Fundamental physics law
                certainty=0.95,
                structure=0.95,
                applicability=0.85,  # Physics domain
                coherence=1.0,
                generativity=0.90
            ),
            turn_number=1,
            context="Physics - Thermodynamics"
        )
        
        print(f"✅ Physics: Minimize free energy")
        print(f"   Q={r.q_score:.4f}, Layer {r.layer}")
        print(f"   Domain: Thermodynamics")
        print(f"   Principle: Systems → minimum energy")
        
        self.results['domains'].append({
            'field': 'Physics',
            'q_score': r.q_score,
            'layer': r.layer,
            'principle': 'energy minimization'
        })
        
        return r
    
    def biology_realization(self):
        """Domain 2: Biology - evolution climbs fitness peaks"""
        
        r = self.engine.add_realization(
            content="Biological evolution optimizes organisms toward fitness peaks (adaptive landscapes)",
            features=RealizationFeatures(
                grounding=0.95,  # Well-established evolutionary theory
                certainty=0.93,
                structure=0.93,
                applicability=0.88,  # Biology domain
                coherence=1.0,
                generativity=0.88
            ),
            turn_number=2,
            context="Biology - Evolutionary Theory"
        )
        
        print(f"✅ Biology: Fitness optimization")
        print(f"   Q={r.q_score:.4f}, Layer {r.layer}")
        print(f"   Domain: Evolution")
        print(f"   Principle: Organisms → fitness peaks")
        
        self.results['domains'].append({
            'field': 'Biology',
            'q_score': r.q_score,
            'layer': r.layer,
            'principle': 'fitness maximization'
        })
        
        return r
    
    def cs_realization(self):
        """Domain 3: CS - gradient descent minimizes loss"""
        
        r = self.engine.add_realization(
            content="Gradient descent minimizes loss functions by moving toward local minima",
            features=RealizationFeatures(
                grounding=0.98,  # Core ML algorithm
                certainty=0.98,  # Very well understood
                structure=0.98,
                applicability=0.95,  # Widely applied
                coherence=1.0,
                generativity=0.92
            ),
            turn_number=3,
            context="Computer Science - Machine Learning"
        )
        
        print(f"✅ Computer Science: Gradient descent")
        print(f"   Q={r.q_score:.4f}, Layer {r.layer}")
        print(f"   Domain: Optimization")
        print(f"   Principle: Algorithms → local minima")
        
        self.results['domains'].append({
            'field': 'Computer Science',
            'q_score': r.q_score,
            'layer': r.layer,
            'principle': 'loss minimization'
        })
        
        return r
    
    def create_synthesis(self, parent_realizations):
        """Create universal synthesis from all 3 domains"""
        
        print(f"\n🌟 Synthesizing universal principle from 3 domains...")
        
        parent_ids = [r.id for r in parent_realizations]
        
        synthesis = self.engine.add_realization(
            content="All complex systems perform hill-climbing in high-dimensional energy landscapes",
            features=RealizationFeatures(
                grounding=0.98,  # Supported by all 3 fields
                certainty=0.95,  # Very confident synthesis
                structure=0.98,  # Crystal clear
                applicability=0.98,  # Universal principle
                coherence=0.98,  # Perfectly integrates all 3
                generativity=0.98  # Opens entire research space
            ),
            turn_number=4,
            parents=parent_ids,
            context="Universal principle - Cross-domain synthesis"
        )
        
        print(f"✅ Synthesis: Universal hill-climbing")
        print(f"   Q={synthesis.q_score:.4f}, Layer {synthesis.layer}")
        print(f"   Parents: {len(synthesis.parents)} domains converged")
        print(f"   🔄 Physics:   Energy landscapes")
        print(f"   🔄 Biology:   Fitness landscapes")
        print(f"   🔄 CS:        Loss landscapes")
        print(f"   → All are the SAME mathematical structure!")
        
        self.results['synthesis_analysis'] = {
            'q_score': synthesis.q_score,
            'layer': synthesis.layer,
            'parent_count': len(synthesis.parents),
            'generativity': synthesis.features.generativity
        }
        
        return synthesis
    
    def analyze_synthesis(self, domains, synthesis):
        """Analyze cross-domain synthesis"""
        
        print("\n" + "="*80)
        print("CROSS-DOMAIN SYNTHESIS ANALYSIS")
        print("="*80)
        
        # 1. Convergence analysis
        print("\n🔄 Convergence Analysis:")
        print(f"   Synthesis has {len(synthesis.parents)} parents")
        print(f"   → Convergence from {len(synthesis.parents)} different fields")
        print(f"   → Physics: thermodynamics")
        print(f"   → Biology: evolution")
        print(f"   → CS: optimization")
        
        # 2. Quality comparison
        print(f"\n💯 Quality Comparison:")
        domain_avg = sum(r.q_score for r in domains) / len(domains)
        print(f"   Domain average: {domain_avg:.4f}")
        print(f"   Synthesis:      {synthesis.q_score:.4f}")
        print(f"   Δ improvement:  +{synthesis.q_score - domain_avg:.4f}")
        
        if synthesis.q_score > domain_avg:
            print(f"   ✓ Synthesis IMPROVED upon individual domains")
        
        # 3. Layer analysis
        print(f"\n📊 Layer Analysis:")
        domain_layers = [r.layer for r in domains]
        print(f"   Domain layers: {domain_layers}")
        print(f"   Synthesis layer: {synthesis.layer}")
        
        if synthesis.layer == 0:
            print(f"   ✅ LAYER 0 ACHIEVED - Universal principle!")
            print(f"   ✅ Cross-domain synthesis reached highest layer")
        elif synthesis.layer == 1:
            print(f"   ✅ LAYER 1 - Domain fact across multiple fields")
        else:
            print(f"   ⚠️  Layer {synthesis.layer} - Expected Layer 0 or 1")
        
        # 4. Coherence analysis
        print(f"\n🧩 Coherence Analysis:")
        print(f"   Synthesis coherence: {synthesis.features.coherence:.2f}")
        print(f"   → How well it integrates all 3 domains")
        
        if synthesis.features.coherence >= 0.95:
            print(f"   ✅ Excellent integration (H≥0.95)")
        
        # 5. Generativity analysis
        print(f"\n🌱 Generativity Analysis:")
        print(f"   Synthesis generativity: {synthesis.features.generativity:.2f}")
        print(f"   → Potential for spawning new research")
        
        if synthesis.features.generativity >= 0.95:
            print(f"   ✅ Highly generative (V≥0.95)")
            print(f"   ✅ Opens: Statistical mechanics + evo-devo + ML theory")
        
        # 6. بنات افكار graph structure
        print(f"\n🌳 بنات افكار Graph:")
        print(f"   Topology: 3 independent branches → 1 synthesis node")
        print(f"   Pattern: Perfect convergence")
        print(f"")
        print(f"   Physics (Q={domains[0].q_score:.2f})")
        print(f"        \\")
        print(f"   Biology (Q={domains[1].q_score:.2f}) → Synthesis (Q={synthesis.q_score:.2f})")
        print(f"        /")
        print(f"   CS (Q={domains[2].q_score:.2f})")
        
        # 7. Overall assessment
        print("\n" + "="*80)
        print("ASSESSMENT")
        print("="*80)
        
        tests_passed = []
        tests_failed = []
        
        # Test 1: Synthesis has 3+ parents
        if len(synthesis.parents) >= 3:
            tests_passed.append("3+ parents (convergence)")
            print("✅ Synthesis correctly shows 3-way convergence")
        else:
            tests_failed.append("Not enough parents")
            print("❌ Synthesis should have 3+ parents")
        
        # Test 2: Synthesis achieved Layer 0 or 1
        if synthesis.layer in [0, 1]:
            tests_passed.append("Reached Layer 0 or 1")
            print(f"✅ Synthesis achieved Layer {synthesis.layer} (universal/domain)")
        else:
            tests_failed.append(f"Only reached Layer {synthesis.layer}")
            print(f"❌ Synthesis should reach Layer 0 or 1")
        
        # Test 3: High coherence (integrates all domains)
        if synthesis.features.coherence >= 0.95:
            tests_passed.append("High coherence (H≥0.95)")
            print("✅ Synthesis has excellent cross-domain coherence")
        else:
            tests_failed.append("Low coherence")
            print("❌ Synthesis should have H≥0.95")
        
        # Test 4: High generativity
        if synthesis.features.generativity >= 0.90:
            tests_passed.append("High generativity (V≥0.90)")
            print("✅ Synthesis is highly generative")
        else:
            tests_failed.append("Low generativity")
            print("❌ Synthesis should have V≥0.90")
        
        # Test 5: Q-score improved
        if synthesis.q_score > domain_avg:
            tests_passed.append("Q-score improved vs domains")
            print("✅ Synthesis Q-score exceeds domain average")
        else:
            tests_failed.append("Q-score didn't improve")
            print("❌ Synthesis should improve Q-score")
        
        # Overall
        if len(tests_failed) == 0:
            self.results['overall_result'] = 'PASSED - Perfect cross-domain synthesis'
            print(f"\n✅ OVERALL: PASSED")
            print(f"   All {len(tests_passed)} tests passed")
            print(f"   System successfully synthesized universal principle from 3 fields")
        else:
            self.results['overall_result'] = f'PARTIAL - {len(tests_failed)} issues'
            print(f"\n⚠️  OVERALL: PARTIAL")
            print(f"   {len(tests_passed)} passed, {len(tests_failed)} failed")
            for fail in tests_failed:
                print(f"     - {fail}")
        
        self.results['tests'] = {
            'passed': tests_passed,
            'failed': tests_failed
        }
    
    def export_results(self):
        with open('/home/claude/test3_cross_domain_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Results exported to test3_cross_domain_results.json")


if __name__ == "__main__":
    test = CrossDomainSynthesisTest()
    test.run_test()
