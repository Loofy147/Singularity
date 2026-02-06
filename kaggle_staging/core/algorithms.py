import numpy as np
from typing import Dict, List, Tuple, Any

class SuffixSmoothing:
    """
    Implements the recursion formula for suffix smoothing by successive abstraction.
    P(t | s) = (1 - lambda) * P_hat(t | s) + lambda * P(t | s')
    """
    def __init__(self, training_data: Dict[str, Dict[str, int]], smoothing_weight: float = 0.5):
        self.data = training_data  # {suffix: {tag: count}}
        self.lmbda = smoothing_weight

    def get_tag_probability(self, tag: str, suffix: str) -> float:
        if not suffix:
            # Base case: unigram probability or uniform
            return 1.0 / 10.0 # Simplified

        counts = self.data.get(suffix, {})
        total = sum(counts.values())

        p_hat = counts.get(tag, 0) / total if total > 0 else 0

        # Recurse with shorter suffix
        p_recursive = self.get_tag_probability(tag, suffix[1:])

        return (1 - self.lmbda) * p_hat + self.lmbda * p_recursive

class EKRLS:
    """
    Simplified Extended Kernel Recursive Least Squares.
    Handles non-linear state estimation using the kernel trick.
    """
    def __init__(self, kernel_sigma: float = 1.0, regularization: float = 0.1):
        self.sigma = kernel_sigma
        self.reg = regularization
        self.dictionary: List[np.ndarray] = []
        self.alpha: List[float] = []
        self.K_inv: Optional[np.ndarray] = None

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * self.sigma**2))

    def update(self, x: np.ndarray, y: float):
        """Update the model with a new measurement (x, y)."""
        if not self.dictionary:
            self.dictionary.append(x)
            self.K_inv = np.array([[1.0 / (self._kernel(x, x) + self.reg)]])
            self.alpha = [y * self.K_inv[0, 0]]
            return

        # Kernel vector k = [k(x1, x), ..., k(xn, x)]
        k = np.array([self._kernel(xi, x) for xi in self.dictionary])

        # Update logic (simplified RLS update in RKHS)
        k_val = self._kernel(x, x)

        # In a real EKRLS, we'd check if x is 'novel' enough to add to dictionary
        # For this demonstration, we just update weights
        self.dictionary.append(x)
        # Expansion of K_inv and alpha would go here
        # ... simplified for demonstration ...
        pass

    def predict(self, x: np.ndarray) -> float:
        if not self.dictionary:
            return 0.0
        k = np.array([self._kernel(xi, x) for xi in self.dictionary])
        return np.dot(k, self.alpha) if len(self.alpha) == len(k) else 0.0

if __name__ == "__main__":
    # Demo Suffix Smoothing
    data = {"ing": {"VBG": 10, "NN": 2}, "ng": {"VBG": 12, "NN": 5}, "g": {"VBG": 20, "NN": 10}}
    smoother = SuffixSmoothing(data)
    print(f"P(VBG | ing) = {smoother.get_tag_probability('VBG', 'ing'):.4f}")

    # Demo EKRLS
    ekrls = EKRLS()
    ekrls.update(np.array([1, 2]), 5.0)
    print(f"EKRLS initialized with 1 sample.")
