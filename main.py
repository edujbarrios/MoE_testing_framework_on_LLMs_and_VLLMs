#!/usr/bin/env python3
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import numpy as np
from text_processing.text_moe import TextMoE
from image_processing.image_moe import ImageMoE
from moe_variants.switched_moe import SwitchedMoE
from utils.visualizer import print_banner, print_separator
from utils.tutorial import run_tutorial
from utils.dashboard import launch_dashboard
from benchmarking import run_benchmark, generate_report

console = Console()

def main():
    print_banner()
    # Para pruebas, directamente ejecutamos la opción 1
    demo_text_moe()

def run_benchmarks():
    print_separator("MoE Benchmarking Suite")

    # Initialize variants
    variants = {
        'TextMoE': TextMoE(num_experts=3),
        'ImageMoE': ImageMoE(num_experts=4),
        'SwitchedMoE': SwitchedMoE(num_experts=3)
    }

    # Run benchmarks for each variant
    for name, variant in variants.items():
        console.print(f"\n[bold cyan]Benchmarking {name}...[/]")
        input_type = 'image' if name == 'ImageMoE' else 'text'
        try:
            results = run_benchmark(variant, input_type=input_type)
            generate_report(results, name)
        except Exception as e:
            console.print(f"[red]Error benchmarking {name}:[/] {str(e)}")

def demo_switched_moe():
    print_separator("Switched MoE Demo")

    # Initialize Switched MoE with adjusted complexity threshold
    switched_moe = SwitchedMoE(num_experts=3, complexity_threshold=0.6)

    # Comprehensive test inputs with varying complexity
    text_inputs = [
        "The cat",                                     # Simple
        "The quick brown fox",                        # Medium
        "Supercalifragilisticexpialidocious!",       # Complex
        "AI and ML are transforming technology",      # Medium-Complex
        "Python programming is fun & efficient @ 2x",  # Complex with special chars,
        "A",                                          # Extremely simple
        "The @ quick # brown & fox * jumped!",        # High special char density
        "Pneumonoultramicroscopicsilicovolcanoconiosis"  # Extreme length
    ]

    console.print("\n[yellow]Testing Switched MoE with text inputs:[/]")

    # Store results for analysis
    complexities = []
    expert_assignments = []

    # Process each input and show detailed results
    for text in text_inputs:
        try:
            result = switched_moe.process(text)
            metrics = switched_moe.get_metrics()
            complexity = switched_moe._compute_complexity(text)

            complexities.append(complexity)
            expert_assignments.append(switched_moe.route(text))

            # Display input and results with rich formatting
            console.print(f"\nInput: [cyan]{text}[/]")
            console.print(f"[green]{result}[/]")

            # Show complexity breakdown
            special_chars = sum(not c.isalnum() and not c.isspace() for c in text) / len(text)
            unique_ratio = len(set(text.lower())) / len(text)

            console.print("[dim]Complexity Breakdown:[/]")
            console.print(f"[dim]- Length Factor: {len(text)/100 * 0.3:.2f}[/]")
            console.print(f"[dim]- Character Variety: {unique_ratio * 0.3:.2f}[/]")
            console.print(f"[dim]- Special Characters: {special_chars * 0.2:.2f}[/]")

            # Show threshold reference
            threshold_info = (
                "[dim]Threshold Reference: "
                f"Simple < {metrics['complexity_threshold']*0.33:.2f} ≤ "
                f"Medium < {metrics['complexity_threshold']*0.66:.2f} ≤ "
                "Complex[/]"
            )
            console.print(threshold_info)

        except Exception as e:
            console.print(f"[red]Error processing input:[/] {str(e)}")

    # Display expert distribution
    console.print("\n[yellow]Expert Assignment Distribution:[/]")
    total_assignments = len(expert_assignments)
    for expert_id in range(switched_moe.num_experts):
        count = expert_assignments.count(expert_id)
        percentage = (count / total_assignments) * 100
        bar_length = int(percentage / 5)
        bar = "█" * bar_length
        console.print(f"Expert {expert_id}: {count} assignments ({percentage:.1f}%)")
        console.print(f"[blue]{bar}[/]")

    # Display complexity range
    console.print("\n[yellow]Complexity Statistics:[/]")
    console.print(f"Minimum Complexity: [cyan]{min(complexities):.2f}[/]")
    console.print(f"Maximum Complexity: [cyan]{max(complexities):.2f}[/]")
    console.print(f"Average Complexity: [cyan]{sum(complexities)/len(complexities):.2f}[/]")

    # Display routing metrics
    console.print("\n[yellow]Routing Metrics Summary:[/]")
    metrics = switched_moe.get_metrics()
    for expert_id, description in metrics['expert_descriptions'].items():
        console.print(f"Expert {expert_id}: [blue]{description}[/]")

def demo_text_moe():
    print_separator("Text MoE Demo")

    # Example text data with varied complexity and patterns
    texts = [
        "The quick brown fox jumps over the lazy dog",  # Classic sentence with mixed lengths
        "AI and ML are transforming technology rapidly",  # Technical terms
        "Python programming is both powerful and elegant",  # Programming related
        "Data scientists analyze complex patterns",  # Domain specific
        "@#$% special ch@r@cters !!! 123",  # Special characters
        "Supercalifragilisticexpialidocious",  # Very long word
        "a b c d e f g",  # Very short words
        "The_quick_brown_fox_1234"  # Mixed with underscores and numbers
    ]

    text_moe = TextMoE(num_experts=3)

    # Track metrics for analysis
    complexity_scores = []
    expert_assignments = {0: 0, 1: 0, 2: 0}
    token_lengths = []

    console.print("\n[bold cyan]Processing Text Samples:[/]")

    for idx, text in enumerate(texts, 1):
        console.print(f"\n[yellow]Sample {idx}:[/] {text}")
        try:
            # Process text
            result = text_moe.process(text)

            # Calculate and store metrics
            complexity = text_moe._compute_complexity(text)
            complexity_scores.append(complexity)

            # Track token assignments
            tokens = text.split()
            for token in tokens:
                expert_weights = text_moe._get_expert_weights(token)
                expert = np.argmax(expert_weights)
                expert_assignments[expert] = expert_assignments.get(expert, 0) + 1
                token_lengths.append(len(token))

            # Display results
            console.print("[green]Expert assignments:[/]")
            console.print(result)

            # Show complexity score
            console.print(f"[dim]Complexity Score: {complexity:.2f}[/]")

        except Exception as e:
            console.print(f"[red]Error processing text:[/] {str(e)}")

    # Display summary statistics
    console.print("\n[bold cyan]Analysis Summary:[/]")

    # Complexity statistics
    avg_complexity = sum(complexity_scores) / len(complexity_scores)
    max_complexity = max(complexity_scores)
    min_complexity = min(complexity_scores)

    console.print("\n[yellow]Complexity Statistics:[/]")
    console.print(f"Average Complexity: [cyan]{avg_complexity:.2f}[/]")
    console.print(f"Maximum Complexity: [cyan]{max_complexity:.2f}[/]")
    console.print(f"Minimum Complexity: [cyan]{min_complexity:.2f}[/]")

    # Token length statistics
    avg_length = sum(token_lengths) / len(token_lengths)
    max_length = max(token_lengths)
    min_length = min(token_lengths)

    console.print("\n[yellow]Token Length Statistics:[/]")
    console.print(f"Average Length: [cyan]{avg_length:.1f}[/] characters")
    console.print(f"Maximum Length: [cyan]{max_length}[/] characters")
    console.print(f"Minimum Length: [cyan]{min_length}[/] characters")

    # Show distribution of expert assignments
    console.print("\n[yellow]Expert Assignment Distribution:[/]")
    total_assignments = sum(expert_assignments.values()) or 1
    expert_descriptions = {
        0: "short text specialist",
        1: "medium text specialist",
        2: "long text specialist"
    }

    for expert_id, description in expert_descriptions.items():
        count = expert_assignments[expert_id]
        percentage = (count / total_assignments) * 100
        bar_length = int(percentage / 5)
        bar = "█" * bar_length
        console.print(f"Expert {expert_id} ({description}):")
        console.print(f"[blue]{bar}[/] ({percentage:.1f}%)")

    # Visual separator
    console.print("\n" + "─" * 50)

def demo_image_moe():
    print_separator("Image MoE Demo")

    try:
        # Generate diverse test images
        image_samples = [
            ("Simple Uniform", np.ones((32, 32)) * 0.5),  # Uniform gray
            ("Gradient", np.linspace(0, 1, 32).reshape(-1, 1) * np.ones((1, 32))),  # Horizontal gradient
            ("Checkerboard", np.indices((32, 32)).sum(axis=0) % 2),  # Checkerboard pattern
            ("Random Noise", np.random.rand(32, 32)),  # Complex noise pattern
            ("Circle Pattern", np.fromfunction(lambda i, j: ((i-16)**2 + (j-16)**2 < 100), (32, 32)).astype(float))  # Circle
        ]

        image_moe = ImageMoE(num_experts=4)
        console.print("\n[bold cyan]Processing various image patterns:[/]")

        # Initialize counters for expert utilization
        expert_counts = {i: 0 for i in range(4)}
        complexity_values = []

        # Process and analyze each image
        for name, image_data in image_samples:
            console.print(f"\n[yellow]Testing: {name}[/]")

            # Process image and get results
            result = image_moe.process(image_data)

            # Display region analysis
            console.print("\n[green]Expert assignments by region:[/]")
            console.print(result)

            # Calculate overall image statistics
            mean_val = np.mean(image_data)
            std_val = np.std(image_data)
            edge_val = np.mean(np.abs(np.diff(image_data)))
            complexity = std_val * 0.4 + edge_val * 0.6

            # Update dashboard metrics
            image_moe.dashboard.update_metrics('image_complexity', complexity)
            complexity_values.append(complexity)

            # Show image statistics
            console.print("\n[cyan]Image Statistics:[/]")
            console.print(f"Mean Intensity: {mean_val:.2f}")
            console.print(f"Standard Deviation: {std_val:.2f}")
            console.print(f"Edge Intensity: {edge_val:.2f}")
            console.print(f"Overall Complexity: {complexity:.2f}")

            # Visual separator
            console.print("\n" + "─" * 50)

        # Show expert utilization summary with visual bars
        expert_desc = {
            0: "dark uniform regions",
            1: "bright uniform regions",
            2: "edge regions",
            3: "complex textures"
        }

        console.print("\n[bold cyan]Expert Specialization Summary:[/]")
        total_assignments = sum(expert_counts.values()) or 1
        for expert_id, description in expert_desc.items():
            count = expert_counts[expert_id]
            percentage = (count / total_assignments) * 100
            bar_length = int(percentage / 5)
            bar = "█" * bar_length
            console.print(f"Expert {expert_id}: [blue]{description}[/]")
            console.print(f"[dim]{count} assignments ({percentage:.1f}%)[/]")
            console.print(f"[blue]{bar}[/]")

        # Show complexity statistics
        console.print("\n[bold cyan]Complexity Analysis:[/]")
        console.print(f"Minimum Complexity: [cyan]{min(complexity_values):.2f}[/]")
        console.print(f"Maximum Complexity: [cyan]{max(complexity_values):.2f}[/]")
        console.print(f"Average Complexity: [cyan]{np.mean(complexity_values):.2f}[/]")

        # Launch interactive dashboard
        console.print("\n[yellow]Launching interactive complexity dashboard...[/]")
        console.print("[dim]Press Ctrl+C in the dashboard window to return to menu[/]")
        image_moe.dashboard.display()

    except Exception as e:
        console.print(f"[red]Error processing images:[/] {str(e)}")
        console.print("[yellow]Traceback:[/]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    main()