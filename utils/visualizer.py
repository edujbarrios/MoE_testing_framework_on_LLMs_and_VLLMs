from rich.console import Console
from rich import box
from rich.panel import Panel

console = Console()

def print_banner():
    banner = """
    ╔══════════════════════════════════════╗
    ║     Mixture of Experts Demo          ║
    ║     Text & Image Processing          ║
    ╚══════════════════════════════════════╝
    """
    console.print(Panel(banner, box=box.DOUBLE))

def print_separator(title):
    separator = f"""
    {'='*50}
    {title}
    {'='*50}
    """
    console.print(separator, style="bold blue")
