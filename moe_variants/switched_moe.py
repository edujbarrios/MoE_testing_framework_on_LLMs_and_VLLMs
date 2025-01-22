import numpy as np
from rich.console import Console
from .base_moe import BaseMoE

class SwitchedMoE(BaseMoE):
    """Switched Mixture of Experts - routes each input to a single expert."""

    def __init__(self, num_experts=3, complexity_threshold=0.5):
        super().__init__(num_experts)
        self.complexity_threshold = complexity_threshold
        self.console = Console()

    def _compute_complexity(self, input_data):
        """Compute input complexity score with improved metrics."""
        if isinstance(input_data, str):
            # Text complexity: based on length, unique chars ratio, and special characters
            length = len(input_data)
            if length == 0:
                return 0.0

            unique_ratio = len(set(input_data.lower())) / length
            special_chars = sum(not c.isalnum() and not c.isspace() for c in input_data) / length
            word_length_avg = sum(len(word) for word in input_data.split()) / len(input_data.split()) if input_data.split() else 0

            complexity = (
                length / 100 * 0.3 +  # Length factor
                unique_ratio * 0.3 +   # Character variety
                special_chars * 0.2 +  # Special character complexity
                word_length_avg / 10 * 0.2  # Average word length
            )

            # Ensure very simple inputs go to expert 0
            if length <= 1 or complexity < 0.2:
                return 0.1  # This will ensure routing to expert 0
            return complexity

        elif isinstance(input_data, np.ndarray):
            # Image complexity: based on variance, edge density, and local patterns
            if input_data.size == 0:
                return 0.0

            variance = np.var(input_data)
            edges = np.mean(np.abs(np.diff(input_data)))
            local_patterns = np.mean(np.abs(input_data - np.mean(input_data)))

            return (
                variance * 0.4 +
                edges * 0.4 +
                local_patterns * 0.2
            )
        else:
            raise ValueError("Unsupported input type")

    def route(self, inputs):
        """Route input to a single expert based on complexity with improved thresholds."""
        complexity = self._compute_complexity(inputs)

        # Adjusted thresholds for better expert distribution
        if complexity < self.complexity_threshold * 0.33:
            return 0  # Simple patterns expert
        elif complexity < self.complexity_threshold * 0.66:
            return 1  # Medium complexity expert
        else:
            return 2  # High complexity expert

    def process(self, inputs):
        """Process input using the switched routing strategy with detailed metrics."""
        chosen_expert = self.route(inputs)
        complexity = self._compute_complexity(inputs)

        # Calculate confidence based on distance from threshold boundaries
        thresholds = [self.complexity_threshold * 0.33, self.complexity_threshold * 0.66]
        distances = [abs(complexity - t) for t in thresholds]
        confidence = 1.0 - min(distances) / self.complexity_threshold

        result = {
            'expert_id': chosen_expert,
            'complexity_score': complexity,
            'confidence': confidence
        }

        # Enhanced expert descriptions with clearer roles
        expert_desc = {
            0: "simple patterns specialist (basic structures)",
            1: "medium complexity specialist (common patterns)",
            2: "high complexity specialist (advanced patterns)"
        }

        # Format output with more detailed metrics
        return (f"Routed to Expert {chosen_expert} ({expert_desc[chosen_expert]}) "
                f"[complexity: {complexity:.2f}, conf: {confidence:.2f}]")

    def get_metrics(self):
        """Return current routing metrics for dashboard integration."""
        return {
            'complexity_threshold': self.complexity_threshold,
            'expert_descriptions': {
                0: "Simple patterns",
                1: "Medium complexity",
                2: "High complexity"
            }
        }