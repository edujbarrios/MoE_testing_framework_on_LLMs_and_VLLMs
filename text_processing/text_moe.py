import numpy as np
from rich.console import Console
from utils.dashboard import ComplexityDashboard
import time

class TextMoE:
    def __init__(self, num_experts=3):
        self.num_experts = num_experts
        self.console = Console()
        self.dashboard = ComplexityDashboard()

    def _compute_complexity(self, token):
        """Compute token complexity score."""
        # Handle empty string case
        if not token:
            return 0.0

        # Complexity based on length and character variety
        length = len(token)
        unique_chars = len(set(token.lower()))
        return (length / 10) * 0.7 + (unique_chars / length) * 0.3

    def _tokenize(self, text):
        """Simple tokenization by splitting on spaces and removing punctuation."""
        # Remove basic punctuation and split
        text = ''.join(c.lower() for c in text if c.isalnum() or c.isspace())
        return text.split()

    def _get_expert_weights(self, token):
        """Compute expert weights for a token based on characteristics."""
        # Enhanced heuristic: consider token length and content
        complexity = self._compute_complexity(token)
        self.dashboard.update_metrics('text_complexity', complexity)

        length = len(token)
        weights = np.zeros(self.num_experts)

        # Expert 0: Short tokens (1-3 chars)
        # Expert 1: Medium tokens (4-6 chars)
        # Expert 2: Long tokens (7+ chars)
        if length <= 3:
            weights[0] = 1.0
        elif length <= 6:
            weights[1] = 1.0
        else:
            weights[2] = 1.0

        return weights

    def process(self, text):
        """Process text using Mixture of Experts."""
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")

        start_time = time.perf_counter()
        tokens = self._tokenize(text)
        if not tokens:
            raise ValueError("No valid tokens found in input text")

        results = []
        for token in tokens:
            expert_weights = self._get_expert_weights(token)
            chosen_expert = np.argmax(expert_weights)
            self.dashboard.update_metrics('expert_assignment', chosen_expert)
            results.append({
                'token': token,
                'expert': chosen_expert,
                'confidence': expert_weights[chosen_expert]
            })

        processing_time = time.perf_counter() - start_time
        self.dashboard.update_metrics('processing_time', processing_time)

        return self._format_results(results)

    def _format_results(self, results):
        """Format results for display with enhanced descriptions."""
        output = []
        expert_descriptions = {
            0: "short word specialist",
            1: "medium word specialist",
            2: "long word specialist"
        }

        for result in results:
            expert_desc = expert_descriptions[result['expert']]
            output.append(
                f"Token: {result['token']} → Expert {result['expert']} ({expert_desc}) "
                f"[conf: {result['confidence']:.2f}]"
            )
        return "\n".join(output)