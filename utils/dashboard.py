import time
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box
import numpy as np
from threading import Lock

console = Console()

class ComplexityDashboard:
    def __init__(self):
        self.layout = Layout()
        self._metrics_lock = Lock()
        self.metrics = {
            'text_complexity': [],
            'image_complexity': [],
            'expert_assignments': {},
            'processing_times': []
        }
        self._setup_layout()
        self._last_update = time.time()
        self._updates_count = 0

    def _setup_layout(self):
        """Initialize the dashboard layout."""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", size=15),
            Layout(name="footer", size=3)
        )

        self.layout["main"].split_row(
            Layout(name="metrics"),
            Layout(name="experts")
        )

    def _create_header(self):
        """Create the dashboard header."""
        return Panel(
            "[bold blue]Model Complexity Dashboard[/]\n"
            f"[dim]Updates: {self._updates_count} | Last update: {time.time() - self._last_update:.1f}s ago[/]",
            box=box.ROUNDED
        )

    def _create_complexity_table(self):
        """Create a table showing current complexity metrics."""
        table = Table(title="Model Complexity Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        table.add_column("Status", justify="right", style="yellow")

        with self._metrics_lock:
            if self.metrics['text_complexity']:
                avg_text = np.mean(self.metrics['text_complexity'])
                table.add_row(
                    "Text Complexity",
                    f"{avg_text:.2f}",
                    "✓ Active"
                )

            if self.metrics['image_complexity']:
                avg_image = np.mean(self.metrics['image_complexity'])
                table.add_row(
                    "Image Complexity",
                    f"{avg_image:.2f}",
                    "✓ Active"
                )

            if self.metrics['processing_times']:
                avg_time = np.mean(self.metrics['processing_times']) * 1000
                table.add_row(
                    "Processing Time (ms)",
                    f"{avg_time:.2f}",
                    "✓ Active"
                )

            if not any([self.metrics['text_complexity'], 
                       self.metrics['image_complexity'], 
                       self.metrics['processing_times']]):
                table.add_row(
                    "[yellow]Waiting for data...[/]",
                    "",
                    "⋯ Pending"
                )

        return table

    def _create_expert_panel(self):
        """Create a panel showing expert assignments."""
        with self._metrics_lock:
            content = []
            total = sum(self.metrics['expert_assignments'].values()) or 1

            if not self.metrics['expert_assignments']:
                content = ["[yellow]Waiting for expert assignments...[/]"]
            else:
                for expert_id, count in sorted(self.metrics['expert_assignments'].items()):
                    percentage = (count / total) * 100
                    bar_length = int(percentage / 5)
                    bar = "█" * bar_length
                    content.append(
                        f"Expert {expert_id}: {count} assignments ({percentage:.1f}%)\n"
                        f"[blue]{bar}[/]"
                    )

        return Panel("\n".join(content), title="Expert Utilization", box=box.ROUNDED)

    def update_metrics(self, metric_type, value):
        """Update dashboard metrics thread-safely."""
        with self._metrics_lock:
            if metric_type in ['text_complexity', 'image_complexity']:
                self.metrics[metric_type].append(value)
                if len(self.metrics[metric_type]) > 100:  # Keep last 100 values
                    self.metrics[metric_type].pop(0)
            elif metric_type == 'expert_assignment':
                self.metrics['expert_assignments'][value] = \
                    self.metrics['expert_assignments'].get(value, 0) + 1
            elif metric_type == 'processing_time':
                self.metrics['processing_times'].append(value)
                if len(self.metrics['processing_times']) > 100:
                    self.metrics['processing_times'].pop(0)

            self._updates_count += 1
            self._last_update = time.time()

    def _create_status_indicator(self):
        """Create a simple status indicator."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            expand=True
        )
        progress.add_task(
            description="[green]Monitoring model complexity[/]" if self._updates_count > 0 
            else "[yellow]Waiting for data...[/]",
            total=None
        )
        return progress

    def display(self):
        """Display the interactive dashboard."""
        try:
            with Live(self.layout, refresh_per_second=4) as live:
                self.layout["header"].update(self._create_header())
                self.layout["footer"].update(self._create_status_indicator())

                while True:
                    self.layout["main"]["metrics"].update(self._create_complexity_table())
                    self.layout["main"]["experts"].update(self._create_expert_panel())
                    self.layout["header"].update(self._create_header())
                    time.sleep(0.25)  # Refresh every 250ms

        except KeyboardInterrupt:
            console.print("\n[green]Dashboard closed successfully![/]")

def launch_dashboard():
    """Launch the interactive complexity dashboard."""
    dashboard = ComplexityDashboard()
    console.print("\n[bold green]Launching Model Complexity Dashboard...[/]")
    console.print("[yellow]Tip: Run text or image processing demos in another terminal to see metrics[/]")
    console.print("[dim]Press Ctrl+C to exit[/]\n")
    dashboard.display()

if __name__ == "__main__":
    launch_dashboard()