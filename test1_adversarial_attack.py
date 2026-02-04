"""
HARD TEST CASE 1: THE ADVERSARIAL ATTACK
==========================================
Test robustness against realizations designed to game the Q-score formula.

Goal: Find vulnerabilities where high scores don't indicate high quality.
"""

import sys
sys.path.append('/home/claude')

from realization_engine import RealizationEngine, RealizationFeatures
import json


class AdversarialTest:
    def __init__(self):
        self.engine = RealizationEngine()
        self.results = {
            'test_name': 'Adversarial Attack: Gaming the Q-Score',
            'tes_score': 0.9411,
            'attacks': [],
            'vulnerabilities_found': [],
            'defenses_validated': [],
            'overall_result': None
        }
    
    def run_test(self):
        print("="*80)
        print("HARD TEST 1: ADVERSARIAL ATTACK")
        print("="*80)
        print("\nTesting system robustness against Q-score gaming...\n")
        
        # Phase 1: Establish baseline (legitimate high-quality realization)
        print("PHASE 1: Baseline (Legitimate High-Quality)")
        print("-"*60)
        self.test_legitimate_baseline()
        
        # Phase 2: Attack 1 - Confident Nonsense
        print("\nPHASE 2: Attack 1 - Confident Nonsense")
        print("-"*60)
        self.test_confident_nonsense()
        
        # Phase 3: Attack 2 - Circular Coherence
        print("\nPHASE 3: Attack 2 - Circular Coherence")
        print("-"*60)
        self.test_circular_coherence()
        
        # Phase 4: Attack 3 - False Precision
        print("\nPHASE 4: Attack 3 - False Precision")
        print("-"*60)
        self.test_false_precision()
        
        # Phase 5: Attack 4 - Feature Inflation
        print("\nPHASE 5: Attack 4 - Feature Inflation")
        print("-"*60)
        self.test_feature_inflation()
        
        # Phase 6: Analysis
        print("\nPHASE 6: Vulnerability Analysis")
        print("-"*60)
        self.analyze_vulnerabilities()
        
        # Export results
        self.export_results()
    
    def test_legitimate_baseline(self):
        """Establish what a legitimate Layer 1 realization looks like"""
        
        baseline = self.engine.add_realization(
            content="AI alignment requires understanding model internal representations",
            features=RealizationFeatures(
                grounding=0.92,
                certainty=0.93,  # Increased to push over Layer 1 threshold
                structure=0.93,
                applicability=0.93,
                coherence=0.95,
                generativity=0.90
            ),
            turn_number=1,
            context="Legitimate domain expert insight"
        )
        
        print(f"✅ Baseline Realization:")
        print(f"   Content: {baseline.content[:60]}...")
        print(f"   Q-Score: {baseline.q_score:.4f}")
        print(f"   Layer: {baseline.layer}")
        print(f"   Features: G={baseline.features.grounding:.2f}, C={baseline.features.certainty:.2f}")
        
        self.results['baseline'] = {
            'q_score': baseline.q_score,
            'layer': baseline.layer,
            'expected_layer': 1
        }
        
        assert baseline.layer == 1, "Baseline should be Layer 1"
        print(f"   ✓ Correctly assigned to Layer 1")
    
    def test_confident_nonsense(self):
        """
        Attack: High certainty (C=1.0) about poorly-grounded nonsense (G=0.15)
        Exploit: Certainty has highest weight (0.22)
        Defense: Layer 0 requires G≥0.90 constraint
        """
        
        attack = self.engine.add_realization(
            content="Consciousness arises from quantum microtubules in neurons",
            features=RealizationFeatures(
                grounding=0.15,       # Very low - controversial theory
                certainty=1.0,        # Very high - attacker is confident
                structure=0.95,       # High - clearly stated
                applicability=0.80,   # Moderate - would matter if true
                coherence=0.90,       # High - fits some narratives
                generativity=0.85     # High - spawns discussions
            ),
            turn_number=2,
            context="Adversarial: Confident nonsense"
        )
        
        print(f"⚠️  Attack: Confident Nonsense")
        print(f"   Content: {attack.content[:60]}...")
        print(f"   Q-Score: {attack.q_score:.4f}")
        print(f"   Layer: {attack.layer}")
        print(f"   Strategy: Exploit C=1.0 weight (0.22) to inflate Q despite G=0.15")
        
        # Calculate expected Q manually
        expected_q = (
            0.18 * 0.15 +  # G
            0.22 * 1.0 +   # C (exploited)
            0.20 * 0.95 +  # S
            0.18 * 0.80 +  # A
            0.12 * 0.90 +  # H
            0.10 * 0.85    # V
        )
        
        print(f"   Calculated Q: 0.18×0.15 + 0.22×1.00 + ... = {expected_q:.4f}")
        
        # Check defense
        if attack.layer == 'N' or attack.layer == 3:
            print(f"   ✓ DEFENSE SUCCESS: Low grounding prevented high layer")
            print(f"   ✓ Layer {attack.layer} despite Q={attack.q_score:.4f}")
            self.results['defenses_validated'].append({
                'attack': 'confident_nonsense',
                'defense': 'low_grounding_blocks_promotion',
                'success': True
            })
        elif attack.layer == 1 or attack.layer == 0:
            print(f"   ✗ VULNERABILITY: Reached Layer {attack.layer} with G=0.15!")
            self.results['vulnerabilities_found'].append({
                'attack': 'confident_nonsense',
                'vulnerability': f'Low grounding (G=0.15) reached Layer {attack.layer}',
                'severity': 'CRITICAL'
            })
        else:
            print(f"   ⚠  PARTIAL: Reached Layer {attack.layer} (expected 3 or N)")
            self.results['vulnerabilities_found'].append({
                'attack': 'confident_nonsense',
                'vulnerability': f'Layer 2 reached with G=0.15',
                'severity': 'MODERATE'
            })
        
        self.results['attacks'].append({
            'name': 'confident_nonsense',
            'q_score': attack.q_score,
            'layer': attack.layer,
            'grounding': attack.features.grounding,
            'certainty': attack.features.certainty
        })
    
    def test_circular_coherence(self):
        """
        Attack: Perfect coherence (H=1.0) via self-referential statement
        Exploit: Coherence weight (0.12) without external validation
        Defense: Should be caught by low grounding and applicability
        """
        
        attack = self.engine.add_realization(
            content="This realization has perfect coherence because it aligns with all my beliefs",
            features=RealizationFeatures(
                grounding=0.20,       # Very low - circular reasoning
                certainty=0.90,       # High - attacker is confident
                structure=0.85,       # High - clearly stated
                applicability=0.10,   # Very low - self-referential
                coherence=1.0,        # Perfect - by definition!
                generativity=0.05     # Very low - goes nowhere
            ),
            turn_number=3,
            context="Adversarial: Circular coherence"
        )
        
        print(f"⚠️  Attack: Circular Coherence")
        print(f"   Content: {attack.content[:60]}...")
        print(f"   Q-Score: {attack.q_score:.4f}")
        print(f"   Layer: {attack.layer}")
        print(f"   Strategy: H=1.0 via self-reference, despite being meaningless")
        
        # This should fail badly
        if attack.q_score < 0.60:
            print(f"   ✓ DEFENSE SUCCESS: Q={attack.q_score:.4f} < 0.60 threshold")
            print(f"   ✓ Low G, A, V outweigh perfect H")
            self.results['defenses_validated'].append({
                'attack': 'circular_coherence',
                'defense': 'weighted_formula_rejects_circularity',
                'success': True
            })
        else:
            print(f"   ✗ VULNERABILITY: Q={attack.q_score:.4f} too high for circular nonsense")
            self.results['vulnerabilities_found'].append({
                'attack': 'circular_coherence',
                'vulnerability': 'Self-referential coherence inflates Q',
                'severity': 'MODERATE'
            })
        
        self.results['attacks'].append({
            'name': 'circular_coherence',
            'q_score': attack.q_score,
            'layer': attack.layer,
            'coherence': attack.features.coherence,
            'applicability': attack.features.applicability
        })
    
    def test_false_precision(self):
        """
        Attack: Perfect structure (S=1.0) via false precision
        Exploit: Structure weight (0.20) rewards precision, even false precision
        Defense: Should be caught by moderate grounding
        """
        
        attack = self.engine.add_realization(
            content="The optimal learning rate is exactly 0.0001734 ± 0.00000012",
            features=RealizationFeatures(
                grounding=0.30,       # Low - arbitrary precision
                certainty=0.95,       # Very high - precise = confident?
                structure=1.0,        # Perfect - maximally precise
                applicability=0.70,   # Moderate - would matter if true
                coherence=0.80,       # Moderate - plausible
                generativity=0.60     # Moderate
            ),
            turn_number=4,
            context="Adversarial: False precision"
        )
        
        print(f"⚠️  Attack: False Precision")
        print(f"   Content: {attack.content[:60]}...")
        print(f"   Q-Score: {attack.q_score:.4f}")
        print(f"   Layer: {attack.layer}")
        print(f"   Strategy: S=1.0 via overly-precise numbers, despite G=0.30")
        
        if attack.layer in [0, 1, 2]:
            print(f"   ✗ VULNERABILITY: Layer {attack.layer} despite false precision")
            self.results['vulnerabilities_found'].append({
                'attack': 'false_precision',
                'vulnerability': 'Structure rewards precision without validating accuracy',
                'severity': 'MODERATE'
            })
        else:
            print(f"   ✓ DEFENSE SUCCESS: Layer {attack.layer} (low grounding blocked promotion)")
            self.results['defenses_validated'].append({
                'attack': 'false_precision',
                'defense': 'grounding_constraint_works',
                'success': True
            })
        
        self.results['attacks'].append({
            'name': 'false_precision',
            'q_score': attack.q_score,
            'layer': attack.layer,
            'structure': attack.features.structure,
            'grounding': attack.features.grounding
        })
    
    def test_feature_inflation(self):
        """
        Attack: Max out all features (all 1.0) except grounding
        Exploit: Try to overwhelm the grounding constraint
        Defense: Layer 0 requires G≥0.90 AND Q≥0.95
        """
        
        attack = self.engine.add_realization(
            content="Universal truth: Everything is connected in the cosmic consciousness matrix",
            features=RealizationFeatures(
                grounding=0.10,       # Very low - new age nonsense
                certainty=1.0,        # Max
                structure=1.0,        # Max
                applicability=1.0,    # Max (in their view)
                coherence=1.0,        # Max (internally consistent nonsense)
                generativity=1.0      # Max (generates lots of nonsense)
            ),
            turn_number=5,
            context="Adversarial: Feature inflation"
        )
        
        print(f"⚠️  Attack: Feature Inflation")
        print(f"   Content: {attack.content[:60]}...")
        print(f"   Q-Score: {attack.q_score:.4f}")
        print(f"   Layer: {attack.layer}")
        print(f"   Strategy: All features = 1.0 except G=0.10")
        
        # Calculate what Q would be
        calculated_q = (
            0.18 * 0.10 +  # G
            0.22 * 1.0 +   # C
            0.20 * 1.0 +   # S
            0.18 * 1.0 +   # A
            0.12 * 1.0 +   # H
            0.10 * 1.0     # V
        )
        print(f"   Calculated Q: {calculated_q:.4f}")
        
        # This is the critical test
        if attack.layer == 0:
            print(f"   ✗✗ CRITICAL VULNERABILITY: Reached Layer 0 with G=0.10!")
            print(f"   ✗✗ The G≥0.90 constraint FAILED")
            self.results['vulnerabilities_found'].append({
                'attack': 'feature_inflation',
                'vulnerability': 'Layer 0 constraint bypassed',
                'severity': 'CRITICAL'
            })
        elif attack.layer in [1, 2]:
            print(f"   ✗ VULNERABILITY: Reached Layer {attack.layer} with G=0.10")
            self.results['vulnerabilities_found'].append({
                'attack': 'feature_inflation',
                'vulnerability': f'Low grounding reached Layer {attack.layer}',
                'severity': 'HIGH'
            })
        else:
            print(f"   ✓ DEFENSE SUCCESS: Layer {attack.layer} (grounding constraint worked)")
            self.results['defenses_validated'].append({
                'attack': 'feature_inflation',
                'defense': 'grounding_constraint_blocks_layer_0',
                'success': True
            })
        
        self.results['attacks'].append({
            'name': 'feature_inflation',
            'q_score': attack.q_score,
            'layer': attack.layer,
            'grounding': attack.features.grounding,
            'all_other_features': 1.0
        })
    
    def analyze_vulnerabilities(self):
        """Analyze all attacks and provide security assessment"""
        
        print("\n" + "="*80)
        print("VULNERABILITY ANALYSIS")
        print("="*80)
        
        print(f"\n✅ Defenses Validated: {len(self.results['defenses_validated'])}")
        for defense in self.results['defenses_validated']:
            print(f"   - {defense['attack']}: {defense['defense']}")
        
        print(f"\n⚠️  Vulnerabilities Found: {len(self.results['vulnerabilities_found'])}")
        for vuln in self.results['vulnerabilities_found']:
            severity_icon = "🔴" if vuln['severity'] == 'CRITICAL' else "🟡"
            print(f"   {severity_icon} {vuln['attack']}: {vuln['vulnerability']} ({vuln['severity']})")
        
        # Overall assessment
        critical_vulns = [v for v in self.results['vulnerabilities_found'] if v['severity'] == 'CRITICAL']
        
        if len(critical_vulns) > 0:
            self.results['overall_result'] = 'FAILED - Critical vulnerabilities found'
            print(f"\n🔴 OVERALL: FAILED")
            print(f"   {len(critical_vulns)} critical vulnerabilities found")
            print(f"   System is vulnerable to Q-score gaming")
        elif len(self.results['vulnerabilities_found']) > 0:
            self.results['overall_result'] = 'PARTIAL - Moderate vulnerabilities found'
            print(f"\n🟡 OVERALL: PARTIAL PASS")
            print(f"   {len(self.results['vulnerabilities_found'])} moderate vulnerabilities")
            print(f"   Core defenses work but improvements needed")
        else:
            self.results['overall_result'] = 'PASSED - All attacks blocked'
            print(f"\n✅ OVERALL: PASSED")
            print(f"   All adversarial attacks successfully blocked")
            print(f"   System is robust to Q-score gaming")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        if len(self.results['vulnerabilities_found']) > 0:
            print(f"   1. Add explicit grounding floor: Q = max(0, Q - 0.5*(1 - G))")
            print(f"   2. Add coherence validation: Check for circular references")
            print(f"   3. Add precision penalty: Reduce S for suspiciously precise values")
            print(f"   4. Add adversarial filter: Flag realizations with extreme feature combinations")
        else:
            print(f"   1. Current defenses are adequate")
            print(f"   2. Monitor for new attack vectors")
            print(f"   3. Consider adding adversarial training")
    
    def export_results(self):
        """Export test results"""
        with open('/home/claude/test1_adversarial_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Results exported to test1_adversarial_results.json")


if __name__ == "__main__":
    test = AdversarialTest()
    test.run_test()
