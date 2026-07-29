import math
from collections import defaultdict

class AdjacentPMI:
    """Computes Negative Pointwise Mutual Information for adjacent tokens.
    Used to detect word-salad context corruption.
    """
    def __init__(self):
        self.unigrams = defaultdict(int)
        self.bigrams = defaultdict(int)
        self.total_u = 0
        self.total_b = 0

    def fit(self, texts: list[str]) -> None:
        for text in texts:
            words = text.split()
            if not words:
                continue
            for w in words:
                self.unigrams[w] += 1
                self.total_u += 1
            for i in range(len(words)-1):
                self.bigrams[(words[i], words[i+1])] += 1
                self.total_b += 1

    def score(self, text: str) -> float:
        """Returns the average Negative PMI. Higher is more anomalous."""
        words = text.split()
        if len(words) < 2:
            return 0.0
        pmi_sum = 0.0
        V = len(self.unigrams) or 1
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            # .get(), NOT [] indexing: reading an unseen key from a
            # defaultdict INSERTS it, growing the vocabulary while scoring and
            # perturbing later scores.
            p_w1 = (self.unigrams.get(w1, 0) + 1) / (self.total_u + V)
            p_w2 = (self.unigrams.get(w2, 0) + 1) / (self.total_u + V)
            p_b = (self.bigrams.get((w1, w2), 0) + 1) / (self.total_b + V * V)
            # Shifted Negative PMI
            pmi_sum -= math.log(max(p_b / (p_w1 * p_w2), 1e-10))
        return pmi_sum / (len(words) - 1)
