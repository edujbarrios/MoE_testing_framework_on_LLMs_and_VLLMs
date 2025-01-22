import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

console = Console()

def run_tutorial():
    """Execute the interactive tutorial mode."""
    print_tutorial_welcome()
    
    sections = [
        explain_moe_basics,
        explain_vit_connection,
        explain_llm_connection,
        explain_vllm_integration,
        interactive_demo
    ]
    
    for section in sections:
        section()
        if not continue_tutorial():
            break
    
    print_tutorial_completion()

def print_tutorial_welcome():
    welcome = """
    ╔══════════════════════════════════════╗
    ║     Tutorial Interactivo MoE         ║
    ║     Aprende mientras exploras        ║
    ╚══════════════════════════════════════╝
    """
    console.print(Panel(welcome, box=box.DOUBLE))

def explain_moe_basics():
    content = """
    [bold cyan]Mixture of Experts (MoE) - Conceptos Básicos[/]
    
    Un MoE es como un equipo de especialistas, donde cada experto:
    • Se especializa en una tarea específica
    • Procesa solo los datos relevantes a su especialidad
    • Contribuye al resultado final según su experiencia
    
    [bold green]Ejemplo:[/]
    En procesamiento de texto:
    • Expert 0: Maneja palabras cortas
    • Expert 1: Procesa palabras medianas
    • Expert 2: Analiza palabras largas
    """
    console.print(Panel(content, title="Fundamentos"))
    time.sleep(2)

def explain_vit_connection():
    content = """
    [bold cyan]Conexión con Vision Transformers (ViT)[/]
    
    Los ViT usan MoE para:
    • Dividir imágenes en regiones (patches)
    • Asignar expertos a diferentes tipos de regiones
    • Procesar características visuales específicas
    
    [bold green]En nuestro demo:[/]
    • Expert 0: Regiones oscuras uniformes
    • Expert 1: Regiones brillantes uniformes
    • Expert 2: Detección de bordes
    • Expert 3: Análisis de texturas
    """
    console.print(Panel(content, title="ViT y MoE"))
    time.sleep(2)

def explain_llm_connection():
    content = """
    [bold cyan]MoE en Large Language Models (LLMs)[/]
    
    Los LLMs implementan MoE para:
    • Procesar diferentes aspectos del lenguaje
    • Manejar vocabulario especializado
    • Optimizar el uso de recursos
    
    [bold green]Ejemplo en nuestro sistema:[/]
    • Especialización por longitud de tokens
    • Asignación dinámica de expertos
    • Puntuaciones de confianza
    """
    console.print(Panel(content, title="LLMs y MoE"))
    time.sleep(2)

def explain_vllm_integration():
    content = """
    [bold cyan]Vision-Language LLMs y MoE[/]
    
    Los VLLMs combinan:
    • Procesamiento visual (como ViT)
    • Comprensión del lenguaje (como LLM)
    • Integración multimodal
    
    [bold green]Demostrado en nuestro proyecto:[/]
    • Procesamiento paralelo de texto e imágenes
    • Expertos especializados por dominio
    • Integración de resultados
    """
    console.print(Panel(content, title="VLLMs y MoE"))
    time.sleep(2)

def interactive_demo():
    content = """
    [bold cyan]Demostración Interactiva[/]
    
    Verás ejemplos prácticos de:
    1. Procesamiento de texto con MoE
    2. Análisis de imágenes con MoE
    3. Integración multimodal
    """
    console.print(Panel(content, title="Demo"))
    choice = Prompt.ask("¿Qué ejemplo te gustaría ver?", choices=["1", "2", "3"])
    # Aquí se implementará la lógica de cada demo

def continue_tutorial():
    return Prompt.ask("\n¿Continuar con el tutorial?", choices=["s", "n"]).lower() == "s"

def print_tutorial_completion():
    completion = """
    ¡Felicitaciones! Has completado el tutorial de MoE.
    Ahora entiendes mejor cómo esta arquitectura es
    fundamental en el desarrollo de ViT, LLMs y VLLMs.
    """
    console.print(Panel(completion, title="¡Tutorial Completado!", border_style="green"))
