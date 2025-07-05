# DeepSentinel Python SDK Documentation

Comprehensive API documentation for the DeepSentinel Python SDK, built with MkDocs and Material theme.

## Overview

This documentation provides:

- **Landing Page** - Clear value proposition and getting started guide
- **Quick Start** - 5-minute integration tutorial
- **Core Concepts** - In-depth explanations of how DeepSentinel works
- **Tutorials** - Step-by-step guides for common use cases
- **Guides** - Best practices and production advice
- **API Reference** - Auto-generated from source code docstrings
- **FAQ** - Common questions and troubleshooting

## Documentation Structure

```
docs/
├── index.md                    # Landing page
├── quickstart.md              # Quick start guide
├── concepts/                  # Core concepts
│   ├── index.md
│   ├── overview.md
│   ├── compliance.md
│   ├── providers.md
│   └── configuration.md
├── tutorials/                 # Step-by-step tutorials
│   ├── index.md
│   └── basic-integration.md
├── guides/                    # Production guides
│   └── index.md
├── faq.md                     # FAQ
├── gen_ref_pages.py          # API reference generator
└── assets/                   # Custom styles and assets
    └── stylesheets/
        └── extra.css
```

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install "deepsentinel[docs]"
```

Or install individual packages:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index
```

### Local Development

Start the development server:

```bash
# Using the build script
python scripts/build-docs.py --serve

# Or directly with mkdocs
mkdocs serve
```

The documentation will be available at http://127.0.0.1:8000

### Building Static Site

Build the static documentation:

```bash
# Using the build script
python scripts/build-docs.py

# Or directly with mkdocs
mkdocs build
```

The built documentation will be in the `site/` directory.

### Clean Build

Clean previous build and rebuild:

```bash
python scripts/build-docs.py --clean
```

## Documentation Features

### Auto-Generated API Reference

The API reference is automatically generated from source code docstrings using mkdocstrings. The `gen_ref_pages.py` script creates documentation pages for all public modules.

**Documented modules:**
- Core client and configuration
- Provider system (OpenAI, Anthropic, registry)
- Compliance engine and policies
- Type definitions and exceptions
- Supporting components (audit, cache, metrics)

### Custom Styling

Custom CSS in `assets/stylesheets/extra.css` provides:
- DeepSentinel brand colors
- Custom admonitions for compliance and provider content
- Enhanced code block styling
- Responsive feature grids
- Accessibility improvements

### Interactive Features

- **Search** - Full-text search across all documentation
- **Dark/Light Mode** - Automatic theme switching
- **Mobile Responsive** - Optimized for all screen sizes
- **Code Copying** - One-click code snippet copying
- **Navigation** - Hierarchical navigation with progress tracking

## Writing Documentation

### Style Guide

Follow these conventions:

1. **Simple Language** - Use clear, accessible English
2. **Code Examples** - Include working code samples
3. **Progressive Disclosure** - Start simple, add complexity gradually
4. **Visual Learning** - Use diagrams and flowcharts where helpful

### Docstring Format

Use Google-style docstrings for API documentation:

```python
def example_function(param1: str, param2: int) -> bool:
    """Short description of the function.
    
    Longer description explaining what the function does
    and when you might use it.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
        
    Example:
        ```python
        result = example_function("test", 42)
        print(result)  # True
        ```
    """
    return param1 == "test" and param2 > 0
```

### Adding New Pages

1. Create markdown file in appropriate directory
2. Add to navigation in `mkdocs.yml`
3. Link from related pages
4. Test locally before committing

### Custom Admonitions

Use custom admonitions for specific content types:

```markdown
!!! compliance "Compliance Note"
    This feature helps ensure GDPR compliance.

!!! provider "Provider Specific"
    This only applies to OpenAI requests.
```

## Deployment

### GitHub Pages

The documentation can be deployed to GitHub Pages:

```bash
mkdocs gh-deploy
```

### Custom Domain

Configure custom domains in `mkdocs.yml`:

```yaml
site_url: https://deepsentinel-ai.github.io/deepsentinel-python
```

### CI/CD Integration

Add to your CI/CD pipeline:

```yaml
- name: Build Documentation
  run: |
    pip install "deepsentinel[docs]"
    mkdocs build

- name: Deploy Documentation  
  run: mkdocs gh-deploy --force
```

## Maintenance

### Updating API Reference

The API reference updates automatically when source code changes. To manually regenerate:

```bash
python docs/gen_ref_pages.py
```

### Adding New Tutorials

1. Create tutorial file in `docs/tutorials/`
2. Add to `docs/tutorials/index.md`
3. Update navigation in `mkdocs.yml`
4. Cross-link from related pages

### Monitoring

Monitor documentation health:

- **Build Status** - Ensure builds succeed
- **Link Checking** - Verify internal links work
- **User Feedback** - Review feedback from users
- **Analytics** - Track popular pages and search terms

## Troubleshooting

### Build Errors

**Missing Dependencies:**
```bash
pip install "deepsentinel[docs]"
```

**Import Errors in API Reference:**
Ensure the `src/deepsentinel` package is in the Python path.

**Mermaid Diagrams Not Rendering:**
Verify PyMdown Extensions is installed and configured.

### Development Server Issues

**Port Already in Use:**
```bash
mkdocs serve --dev-addr 127.0.0.1:8001
```

**Auto-reload Not Working:**
Check file permissions and ensure you're in the correct directory.

### Styling Issues

**Custom CSS Not Loading:**
Verify `extra.css` path in `mkdocs.yml` and file exists.

**Theme Not Applied:**
Check Material theme is installed: `pip install mkdocs-material`

## Contributing

To contribute to the documentation:

1. **Report Issues** - File issues for unclear or missing docs
2. **Suggest Improvements** - Propose better explanations or examples
3. **Submit PRs** - Add new content or fix existing issues
4. **Review Changes** - Help review documentation updates

### Content Guidelines

- **Accuracy** - Ensure code examples work
- **Completeness** - Cover all important use cases
- **Clarity** - Write for global audience
- **Consistency** - Follow established patterns

## Support

For documentation-related questions:

- **GitHub Issues** - Report documentation bugs
- **Discord** - Ask questions in #documentation channel
- **Email** - Contact docs@deepsentinel.ai

---

Built with ❤️ using [MkDocs](https://mkdocs.org) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)