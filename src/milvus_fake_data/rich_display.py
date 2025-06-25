"""Rich terminal display utilities for schema information."""

from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


def display_schema_list(schemas: Dict[str, Dict[str, Any]], title: str) -> None:
    """Display a list of schemas with rich formatting.
    
    Args:
        schemas: Dictionary of schema_id -> schema_info
        title: Title for the display
    """
    console = Console()
    
    if not schemas:
        console.print(f"[yellow]No {title.lower()} found.[/yellow]")
        return
    
    console.print(f"\n[bold blue]{title}[/bold blue]\n")
    
    for schema_id, info in schemas.items():
        # Create a panel for each schema
        content = []
        content.append(f"[dim]{info['description']}[/dim]")
        
        # Add metadata
        metadata_parts = []
        metadata_parts.append(f"Fields: [cyan]{info['fields_count']}[/cyan]")
        
        if info['vector_dims']:
            dims_str = ', '.join(map(str, info['vector_dims']))
            metadata_parts.append(f"Vector dims: [green]{dims_str}[/green]")
        
        content.append(" • ".join(metadata_parts))
        
        # Add use cases if available
        if info.get('use_cases'):
            use_cases_str = ', '.join(info['use_cases'])
            content.append(f"[italic]Use cases: {use_cases_str}[/italic]")
        
        # Create panel
        panel_content = "\n".join(content)
        panel = Panel(
            panel_content,
            title=f"[bold]{schema_id}[/bold] - {info['name']}",
            border_style="bright_blue" if info.get('type') == 'custom' else "green",
            width=80
        )
        console.print(panel)


def display_schema_details(schema_id: str, info: Dict[str, Any], schema_data: Dict[str, Any], 
                          is_builtin: bool) -> None:
    """Display detailed schema information with rich formatting.
    
    Args:
        schema_id: Schema identifier
        info: Schema metadata information
        schema_data: Full schema data
        is_builtin: Whether the schema is built-in
    """
    console = Console()
    
    # Header
    schema_type = "Built-in" if is_builtin else "Custom"
    type_color = "green" if is_builtin else "bright_blue"
    
    console.print(f"\n[bold {type_color}]Schema: {schema_id}[/bold {type_color}] [dim]({schema_type})[/dim]\n")
    
    # Basic information table
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Property", style="bold")
    info_table.add_column("Value")
    
    info_table.add_row("Name", info['name'])
    info_table.add_row("Description", info['description'] or "[dim]No description[/dim]")
    info_table.add_row("Collection", schema_data.get('collection_name', 'N/A'))
    info_table.add_row("Fields", str(info['fields_count']))
    
    if info['vector_dims']:
        dims_str = ', '.join(map(str, info['vector_dims']))
        info_table.add_row("Vector dimensions", f"[green]{dims_str}[/green]")
    
    if info.get('use_cases'):
        use_cases_str = ', '.join(info['use_cases'])
        info_table.add_row("Use cases", f"[italic]{use_cases_str}[/italic]")
    
    console.print(Panel(info_table, title="[bold]Schema Information[/bold]", border_style=type_color))
    
    # Fields table
    fields_table = Table(title="Fields", show_header=True, header_style="bold magenta")
    fields_table.add_column("Name", style="cyan", no_wrap=True)
    fields_table.add_column("Type", style="yellow")
    fields_table.add_column("Properties", style="dim")
    fields_table.add_column("Description", style="dim", max_width=40)
    
    for field in schema_data.get('fields', []):
        field_name = field['name']
        field_type = field['type']
        
        # Build properties string
        properties = []
        if field.get('is_primary'):
            properties.append("[bold red]PRIMARY[/bold red]")
        if field.get('auto_id'):
            properties.append("[blue]AUTO_ID[/blue]")
        if field.get('nullable'):
            properties.append("[yellow]NULLABLE[/yellow]")
        if 'dim' in field:
            properties.append(f"[green]dim={field['dim']}[/green]")
        if 'max_length' in field:
            properties.append(f"[cyan]max_length={field['max_length']}[/cyan]")
        if 'min' in field or 'max' in field:
            range_info = []
            if 'min' in field:
                range_info.append(f"min={field['min']}")
            if 'max' in field:
                range_info.append(f"max={field['max']}")
            properties.append(f"[dim]{', '.join(range_info)}[/dim]")
        
        properties_str = " ".join(properties) if properties else ""
        description = field.get('description', '')
        
        fields_table.add_row(field_name, field_type, properties_str, description)
    
    console.print(fields_table)
    
    # Usage example
    usage_text = Text()
    usage_text.append("Usage: ", style="bold")
    if is_builtin:
        usage_text.append(f"milvus-fake-data --builtin {schema_id} --rows 1000", style="code")
    else:
        usage_text.append(f"milvus-fake-data --builtin {schema_id} --rows 1000", style="code")
    
    console.print(Panel(usage_text, title="[bold]Usage Example[/bold]", border_style="bright_green"))


def display_error(message: str, details: str = "") -> None:
    """Display error message with rich formatting.
    
    Args:
        message: Main error message
        details: Additional details
    """
    console = Console()
    
    error_text = f"[bold red]✗[/bold red] {message}"
    if details:
        error_text += f"\n[dim]{details}[/dim]"
    
    console.print(Panel(error_text, border_style="red", title="[bold red]Error[/bold red]"))


def display_success(message: str, details: str = "") -> None:
    """Display success message with rich formatting.
    
    Args:
        message: Main success message
        details: Additional details
    """
    console = Console()
    
    success_text = f"[bold green]✓[/bold green] {message}"
    if details:
        success_text += f"\n[dim]{details}[/dim]"
    
    console.print(Panel(success_text, border_style="green", title="[bold green]Success[/bold green]"))


def display_schema_validation(schema_id: str, validation_info: Dict[str, Any]) -> None:
    """Display schema validation results with rich formatting.
    
    Args:
        schema_id: Schema identifier
        validation_info: Validation result information
    """
    console = Console()
    
    console.print(f"\n[bold green]✓ Schema '{schema_id}' is valid![/bold green]\n")
    
    # Create validation summary
    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Property", style="bold")
    summary_table.add_column("Value", style="cyan")
    
    if 'collection_name' in validation_info:
        summary_table.add_row("Collection", validation_info['collection_name'] or 'unnamed')
    
    if 'fields_count' in validation_info:
        summary_table.add_row("Fields found", str(validation_info['fields_count']))
    
    console.print(Panel(summary_table, title="[bold]Validation Summary[/bold]", border_style="green"))
    
    # Display fields if available
    if 'fields' in validation_info:
        fields_tree = Tree("[bold]Fields:[/bold]")
        for field in validation_info['fields']:
            field_name = field.get('name', 'unknown')
            field_type = field.get('type', 'unknown')
            is_primary = field.get('is_primary', False)
            
            field_label = f"[cyan]{field_name}[/cyan]: [yellow]{field_type}[/yellow]"
            if is_primary:
                field_label += " [bold red](PRIMARY)[/bold red]"
            
            fields_tree.add(field_label)
        
        console.print(fields_tree)