import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MetacognitiveObservation:
    iteration: int
    confidence: float
    detected_biases: List[str]
    risk_level: str
    recommendation: str

class MetacognitiveMonitor:
    """
    Supervisory logic that observes the worker process of reasoning.
    Implements self-recursive metacognition.
    """
    def __init__(self):
        self.history: List[MetacognitiveObservation] = []
        self.bias_patterns = {
            'Confirmation Bias': 'seeking only supporting evidence',
            'Availability Bias': 'reliance on vivid/recent examples',
            'Anchoring': 'over-reliance on initial information',
            'Circular Reasoning': 'premise assumes the conclusion'
        }

    def monitor_step(self, iteration: int, reasoning_step: str, confidence: float) -> MetacognitiveObservation:
        detected = []

        # Heuristic bias detection
        if "because it is" in reasoning_step.lower():
            detected.append("Circular Reasoning")

        if iteration == 0 and confidence > 0.95:
            detected.append("Anchoring")

        if "always" in reasoning_step.lower() or "never" in reasoning_step.lower():
            detected.append("Confirmation Bias")

        # Confidence calibration
        risk_level = "LOW"
        recommendation = "Proceed"

        if confidence > 0.9 and iteration < 2:
            risk_level = "HIGH"
            recommendation = "Overconfidence risk detected. Review underlying evidence."
        elif detected:
            risk_level = "MEDIUM"
            recommendation = f"Mitigate {', '.join(detected)} by searching for disconfirming evidence."

        obs = MetacognitiveObservation(
            iteration=iteration,
            confidence=confidence,
            detected_biases=detected,
            risk_level=risk_level,
            recommendation=recommendation
        )
        self.history.append(obs)
        return obs

    def print_audit_trail(self):
        print("\n" + "🔍" * 20)
        print("METACOGNITIVE AUDIT TRAIL")
        print("🔍" * 20)
        for obs in self.history:
            print(f"Iter {obs.iteration} | Conf: {obs.confidence:.2f} | Risk: {obs.risk_level}")
            if obs.detected_biases:
                print(f"  Biases: {', '.join(obs.detected_biases)}")
            print(f"  Rec: {obs.recommendation}")
        print("🔍" * 20 + "\n")

if __name__ == "__main__":
    monitor = MetacognitiveMonitor()
    monitor.monitor_step(0, "I am sure about this because it is obviously true.", 0.99)
    monitor.monitor_step(1, "Always assume the first premise is correct.", 0.85)
    monitor.print_audit_trail()
