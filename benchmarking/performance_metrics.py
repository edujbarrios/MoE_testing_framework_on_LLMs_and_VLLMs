import time
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from pathlib import Path

console = Console()

class MoEBenchmark:
    def __init__(self, moe_variant, num_iterations=100):
        self.moe_variant = moe_variant
        self.num_iterations = num_iterations
        self.results = {
            'execution_times': [],
            'routing_decisions': [],
            'expert_usage': {},
            'complexity_scores': []
        }

    def _generate_text_sample(self, complexity='medium'):
        """Generate text samples of varying complexity."""
        samples = {
            'simple': ['The', 'cat', 'dog', 'ran'],
            'medium': ['Python', 'programming', 'language'],
            'complex': ['Supercalifragilisticexpialidocious', 'Incomprehensibilities']
        }
        return np.random.choice(samples[complexity])

    def _generate_image_sample(self, complexity='medium'):
        """Generate synthetic image samples of varying complexity."""
        size = 32
        if complexity == 'simple':
            # Uniform image
            return np.ones((size, size)) * 0.5
        elif complexity == 'medium':
            # Simple pattern
            img = np.zeros((size, size))
            img[size//4:3*size//4, size//4:3*size//4] = 1.0
            return img
        else:
            # Complex pattern
            return np.random.rand(size, size)

    def run_single_benchmark(self, input_type='text', complexity='medium'):
        """Run a single benchmark iteration."""
        if input_type == 'text':
            sample = self._generate_text_sample(complexity)
        else:
            sample = self._generate_image_sample(complexity)

        start_time = time.perf_counter()
        result = self.moe_variant.process(sample)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        self.results['execution_times'].append(execution_time)
        self.results['complexity_scores'].append(complexity)

        # Track expert usage if available
        if hasattr(self.moe_variant, 'route'):
            expert = self.moe_variant.route(sample)
            self.results['routing_decisions'].append(expert)
            self.results['expert_usage'][expert] = self.results['expert_usage'].get(expert, 0) + 1

        return execution_time

    def run_full_benchmark(self, input_type='text'):
        """Run complete benchmark suite."""
        complexities = ['simple', 'medium', 'complex']
        for complexity in complexities:
            console.print(f"\n[yellow]Running {complexity} complexity benchmarks...[/]")
            for _ in range(self.num_iterations // len(complexities)):
                self.run_single_benchmark(input_type, complexity)

def run_benchmark(moe_variant, input_type='text', num_iterations=100):
    """Run benchmarks for a given MoE variant."""
    benchmark = MoEBenchmark(moe_variant, num_iterations)
    benchmark.run_full_benchmark(input_type)
    return benchmark.results

def plot_execution_times(results, variant_name, save_path):
    """Generate and save execution time distribution plot."""
    plt.figure(figsize=(10, 6))
    plt.hist(np.array(results['execution_times']) * 1000, bins=30, alpha=0.7)
    plt.title(f'Execution Time Distribution - {variant_name}')
    plt.xlabel('Execution Time (ms)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(save_path) / f'{variant_name.lower()}_execution_times.png')
    plt.close()

def plot_expert_usage(results, variant_name, save_path):
    """Generate and save expert usage pie chart."""
    if results['expert_usage']:
        plt.figure(figsize=(8, 8))
        experts = list(results['expert_usage'].keys())
        usage = list(results['expert_usage'].values())
        plt.pie(usage, labels=[f'Expert {e}' for e in experts], autopct='%1.1f%%')
        plt.title(f'Expert Utilization - {variant_name}')
        plt.savefig(Path(save_path) / f'{variant_name.lower()}_expert_usage.png')
        plt.close()

def plot_complexity_performance(results, variant_name, save_path):
    """Generate and save complexity vs performance plot."""
    if results['complexity_scores']:
        plt.figure(figsize=(10, 6))
        times_by_complexity = {
            'simple': [],
            'medium': [],
            'complex': []
        }

        for time, complexity in zip(results['execution_times'], results['complexity_scores']):
            times_by_complexity[complexity].append(time * 1000)  # Convert to ms

        plt.boxplot([times_by_complexity[c] for c in ['simple', 'medium', 'complex']], 
                   labels=['Simple', 'Medium', 'Complex'])
        plt.title(f'Performance by Complexity - {variant_name}')
        plt.ylabel('Execution Time (ms)')
        plt.grid(True, alpha=0.3)
        plt.savefig(Path(save_path) / f'{variant_name.lower()}_complexity_performance.png')
        plt.close()

def generate_report(results, variant_name, save_path="results/benchmarks"):
    """Generate and save benchmark report with visualizations."""
    # Create results directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    avg_time = np.mean(results['execution_times']) * 1000  # Convert to ms
    std_time = np.std(results['execution_times']) * 1000
    p95_time = np.percentile(results['execution_times'], 95) * 1000

    # Generate visualizations
    plot_execution_times(results, variant_name, save_path)
    plot_expert_usage(results, variant_name, save_path)
    plot_complexity_performance(results, variant_name, save_path)

    # Create performance table
    table = Table(title=f"Performance Metrics: {variant_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Average Time (ms)", f"{avg_time:.2f}")
    table.add_row("Std Deviation (ms)", f"{std_time:.2f}")
    table.add_row("95th Percentile (ms)", f"{p95_time:.2f}")

    # Expert utilization if available
    if results['expert_usage']:
        table.add_section()
        total_calls = sum(results['expert_usage'].values())  # Calculate total_calls here
        for expert, count in results['expert_usage'].items():
            percentage = (count / total_calls) * 100
            table.add_row(f"Expert {expert} Usage", f"{percentage:.1f}%")

    # Save and display results
    console.print(table)

    # Save detailed results to file
    report_path = Path(save_path) / f"{variant_name.lower()}_benchmark.txt"
    with open(report_path, 'w') as f:
        f.write(f"Performance Report for {variant_name}\n")
        f.write("="* 50 + "\n\n")
        f.write(f"Number of iterations: {len(results['execution_times'])}\n")
        f.write(f"Average execution time: {avg_time:.2f} ms\n")
        f.write(f"Standard deviation: {std_time:.2f} ms\n")
        f.write(f"95th percentile: {p95_time:.2f} ms\n\n")

        if results['expert_usage']:
            total_calls = sum(results['expert_usage'].values())  # Calculate total_calls here too
            f.write("Expert Utilization:\n")
            for expert, count in results['expert_usage'].items():
                percentage = (count / total_calls) * 100
                f.write(f"Expert {expert}: {percentage:.1f}%\n")

    console.print(f"\n[green]Detailed report and visualizations saved to: {save_path}[/]")