"""Generate API reference pages from source code."""

from pathlib import Path
import mkdocs_gen_files

# Configuration
nav = mkdocs_gen_files.Nav()
src_root = Path("src")
doc_root = Path("reference")

# Modules to document
modules_to_document = [
    "deepsentinel",
    "deepsentinel.client", 
    "deepsentinel.config",
    "deepsentinel.exceptions",
    "deepsentinel.interfaces",
    "deepsentinel.types",
    "deepsentinel.providers",
    "deepsentinel.providers.openai",
    "deepsentinel.providers.anthropic",
    "deepsentinel.providers.registry",
    "deepsentinel.compliance",
    "deepsentinel.compliance.engine",
    "deepsentinel.compliance.policies",
    "deepsentinel.compliance.interceptor",
    "deepsentinel.compliance.detection",
    "deepsentinel.audit",
    "deepsentinel.cache",
    "deepsentinel.metrics",
    "deepsentinel.middleware",
    "deepsentinel.api",
]

def generate_module_page(module_name: str, file_path: Path) -> None:
    """Generate a documentation page for a module."""
    with mkdocs_gen_files.open(file_path, "w") as f:
        # Write module header
        f.write(f"# {module_name}\n\n")
        
        # Add module docstring and API documentation
        f.write(f"::: {module_name}\n")
        f.write("    options:\n")
        f.write("      show_root_heading: true\n")
        f.write("      show_root_toc_entry: true\n")
        f.write("      show_source: false\n")
        f.write("      members_order: source\n")
        f.write("      group_by_category: true\n")
        f.write("      show_category_heading: true\n")
        f.write("      show_if_no_docstring: false\n")
        f.write("      inherited_members: false\n")
        f.write("      filters:\n")
        f.write('        - "!^_"\n')
        f.write('        - "!^__(?!(init|call|str|repr)__)"\n')

# Generate main API reference index
with mkdocs_gen_files.open(doc_root / "index.md", "w") as f:
    f.write("""# API Reference

Complete API documentation for the DeepSentinel Python SDK, automatically generated from source code docstrings.

## Core Components

### [SentinelClient](client.md)
The main client class for interacting with LLM providers through compliance middleware.

### [Configuration](config.md) 
Configuration classes for setting up compliance policies, provider settings, and SDK behavior.

### [Types & Models](types.md)
Data models and type definitions used throughout the SDK.

### [Exceptions](exceptions.md)
Exception hierarchy for handling various error conditions.

### [Interfaces](interfaces.md)
Interface definitions for different LLM operations (chat, completions, embeddings, etc.).

## Provider System

### [Provider Registry](providers/registry.md)
Central registry for managing multiple LLM provider connections.

### [OpenAI Provider](providers/openai.md)
OpenAI-specific provider adapter with full API compatibility.

### [Anthropic Provider](providers/anthropic.md) 
Anthropic Claude provider adapter for accessing Claude models.

### [Base Provider](providers.md)
Base provider interface and common provider functionality.

## Compliance System

### [Compliance Engine](compliance/engine.md)
Core compliance checking and policy enforcement system.

### [Compliance Policies](compliance/policies.md)
Policy management for defining compliance rules and actions.

### [Compliance Interceptor](compliance/interceptor.md)
Request/response interceptor for applying compliance checks.

### [Detection Engines](compliance/detection.md)
Specialized detection engines for PII, PHI, PCI, and custom patterns.

## Supporting Components

### [Audit System](audit.md)
Comprehensive audit logging and compliance tracking.

### [Caching](cache.md)
Performance optimization through intelligent caching of compliance decisions.

### [Metrics](metrics.md)
Performance and compliance metrics collection and reporting.

### [Middleware](middleware.md)
Core middleware infrastructure for request/response processing.

### [API Client](api.md)
HTTP client for communicating with DeepSentinel cloud services.

## Module Index

All SDK modules are documented with their public APIs, including classes, functions, and exceptions:

""")

# Generate documentation for each module
for module_name in modules_to_document:
    # Convert module name to file path
    parts = module_name.split(".")
    
    if len(parts) == 1:
        # Top-level module
        file_path = doc_root / f"{parts[0]}.md"
        nav_parts = [parts[0]]
    else:
        # Nested module
        if len(parts) == 2:
            file_path = doc_root / parts[1] / "index.md" if parts[1] in ['providers', 'compliance'] else doc_root / f"{parts[1]}.md"
            nav_parts = parts[1:]
        else:
            # Three or more parts - create subdirectory structure
            file_path = doc_root / "/".join(parts[1:-1]) / f"{parts[-1]}.md"
            nav_parts = parts[1:]
    
    # Generate the module documentation page
    generate_module_page(module_name, file_path)
    
    # Add to navigation
    nav[nav_parts] = file_path.as_posix()

# Generate SUMMARY.md for navigation
with mkdocs_gen_files.open(doc_root / "SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

print("✅ API reference pages generated successfully")