import numpy as np
import time
from rich.console import Console
from utils.dashboard import ComplexityDashboard

class ImageMoE:
    def __init__(self, num_experts=4):
        self.num_experts = num_experts
        self.console = Console()
        self.dashboard = ComplexityDashboard()

    def _compute_complexity(self, region):
        """Compute region complexity score."""
        mean_val = np.mean(region)
        std_val = np.std(region)
        edge_val = np.mean(np.abs(np.diff(region)))
        complexity = (std_val * 0.4 + edge_val * 0.6)
        self.dashboard.update_metrics('image_complexity', complexity)
        return complexity

    def _get_region_features(self, region):
        """Extract features from image region with validation."""
        if region.size == 0:
            raise ValueError("Empty region provided")

        return {
            'mean': np.mean(region),
            'std': np.std(region),
            'max': np.max(region),
            'min': np.min(region)
        }

    def _get_expert_weights(self, features):
        """Compute expert weights based on region features."""
        weights = np.zeros(self.num_experts)
        #The edited code had a bug where it tried to use region before it was defined in this scope.
        #This line is added to fix the bug.
        mean_val = features['mean']
        std_val = features['std']

        # Enhanced expert assignment based on intensity patterns

        if std_val < 0.1:  # Uniform regions
            if mean_val < 0.5:
                weights[0] = 1.0  # Dark uniform
            else:
                weights[1] = 1.0  # Bright uniform
        elif std_val < 0.3:
            weights[2] = 1.0  # Edge regions
        else:
            weights[3] = 1.0  # Complex texture

        return weights

    def process(self, image):
        """Process image using Mixture of Experts."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if image.ndim != 2:
            raise ValueError("Input must be a 2D grayscale image")

        start_time = time.perf_counter()
        height, width = image.shape
        results = []

        # Process image in 8x8 regions
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                region = image[i:min(i+8, height), j:min(j+8, width)]
                features = self._get_region_features(region)
                expert_weights = self._get_expert_weights(features)
                chosen_expert = np.argmax(expert_weights)
                self.dashboard.update_metrics('expert_assignment', chosen_expert)

                results.append({
                    'region': (i, j),
                    'expert': chosen_expert,
                    'features': features,
                    'confidence': expert_weights[chosen_expert]
                })

        processing_time = time.perf_counter() - start_time
        self.dashboard.update_metrics('processing_time', processing_time)

        return self._format_results(results)

    def _format_results(self, results):
        """Format results for display with enhanced descriptions."""
        expert_descriptions = {
            0: "dark uniform specialist",
            1: "bright uniform specialist",
            2: "edge detection specialist",
            3: "texture analysis specialist"
        }

        output = []
        for result in results:
            expert_desc = expert_descriptions[result['expert']]
            output.append(
                f"Region {result['region']} → Expert {result['expert']} ({expert_desc}) "
                f"[mean: {result['features']['mean']:.2f}, std: {result['features']['std']:.2f}] "
                f"[conf: {result['confidence']:.2f}]"
            )
        return "\n".join(output)