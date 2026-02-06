import sys
import os
import pytest

# Add root to path
sys.path.append(os.getcwd())
from core.omega import OMEGAOrchestrator

def test_omega_cycle():
    """Verify that OMEGA Orchestrator can run a full cycle."""
    omega = OMEGAOrchestrator()
    inputs = [
        "Testing the integration of the realization system.",
        "Ensuring all 13 dimensions are active and contributing."
    ]
    results = omega.run_cycle(inputs, target_q=0.70)

    assert len(results) == len(inputs)
    for q in results:
        assert q > 0.5  # Base Q-score is 0.5, so it should improve

    assert omega.improver.level == 2  # Level should have incremented

def test_engine_layer_assignment():
    """Verify that the engine assigns layers correctly."""
    from core.engine import RealizationEngine, RealizationFeatures
    engine = RealizationEngine()

    # Test Layer 0 (Universal)
    f0 = RealizationFeatures.from_uqs(0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    r0 = engine.add_realization("Universal Truth", f0, 1)
    assert r0.layer == 0

    # Test Layer N (Ephemeral)
    fn = RealizationFeatures.from_uqs(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    rn = engine.add_realization("Ephemeral thought", fn, 1)
    assert rn.layer == 'N'
