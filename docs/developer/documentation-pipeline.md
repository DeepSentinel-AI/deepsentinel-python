# Documentation Build Pipeline

This guide explains how to use and maintain the documentation build pipeline for the DeepSentinel Python SDK.

## Overview

The documentation pipeline consists of:

1. **GitHub Actions Workflow** - Automates building and deploying documentation
2. **Local Build Scripts** - For development and testing
3. **MkDocs Configuration** - Controls documentation structure and appearance
4. **Versioning System** - Manages multiple documentation versions

## GitHub Actions Workflow

The documentation is automatically built and deployed using GitHub Actions. The workflow is defined in `.github/workflows/docs.yml` in the public repository.

### When it runs:

- On every push to the `main` branch (that changes documentation files)
- On pull requests targeting the `main` branch (to verify documentation builds correctly)

### What it does:

1. Sets up Python environment
2. Installs dependencies
3. Checks for broken links in documentation
4. Builds documentation using MkDocs
5. For `main` branch pushes, deploys to GitHub Pages
6. For tagged releases, creates versioned documentation

### Maintenance tasks:

- If you add new Python dependencies for documentation, add them to the workflow file
- If you change the documentation structure, make sure the workflow's validation steps still pass
- Remember that this workflow runs in the public repository context after the git subtree push

## Local Documentation Development

For local development, use the enhanced `build-docs.py` script in the `python/scripts` directory.

### Basic usage:

```bash
# Build documentation
python scripts/build-docs.py

# Start development server
python scripts/build-docs.py --serve

# Start development server and open browser
python scripts/build-docs.py --serve --open

# Clean previous build and rebuild
python scripts/build-docs.py --clean

# Check for broken links
python scripts/build-docs.py --check-links

# Validate documentation structure
python scripts/build-docs.py --validate

# Deploy documentation to GitHub Pages (maintainers only)
python scripts/build-docs.py --deploy --version 1.2.3
```

### Script features:

- **Development server**: Live preview with auto-reload
- **Link checking**: Validates all links are working
- **Validation**: Ensures documentation structure is correct
- **Deployment**: Direct deployment to GitHub Pages with versioning

## MkDocs Configuration

The documentation is built using MkDocs with the Material theme. Configuration is in `mkdocs.yml` (in the root of the public repository).

### Key settings:

- **Theme configuration**: Controls appearance and features
- **Plugins**: Extends MkDocs functionality (API reference generation, navigation)
- **Markdown extensions**: Adds features like admonitions, tabs, code highlighting
- **Navigation**: Defines the documentation structure in the sidebar

### Versioning:

Documentation versioning uses the `mike` tool which is configured in the `extra.version` section. When deploying:

- The latest release is always available as "latest"
- Development documentation is deployed as "dev"
- Each release version gets its own version selector entry

## Adding New Documentation

1. Create new Markdown files in the appropriate directory under `docs/`
2. If creating a new section, add it to the navigation in `mkdocs.yml`
3. Run `python scripts/build-docs.py --serve` to preview changes
4. Run `python scripts/build-docs.py --check-links` to verify links work

## Deploying New Versions

When releasing a new version:

1. Update version numbers in code
2. Push changes to GitHub
3. Create a new release tag
4. The GitHub Actions workflow will automatically deploy the new version

### Manual deployment:

```bash
python scripts/build-docs.py --deploy --version X.Y.Z
```

## Troubleshooting

### Build fails with missing dependencies:

```bash
pip install -e ".[dev]"
pip install mkdocs-material mkdocstrings[python] mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index
```

### Link checking fails:

- Check for typos in URLs
- Ensure referenced files exist
- Verify anchor links (#section-ids) point to actual sections

### Deployment issues:

- Ensure you have proper GitHub permissions
- Check GitHub Actions logs for errors
- Verify GitHub Pages is enabled for the repository

## Public Repository Workflow

The documentation pipeline is configured to work within the public repository after the git subtree push from the main development repository:

1. Changes are made in the main repository under the `python/docs/` directory
2. Changes are pushed to the public repository using git subtree:
   ```bash
   git subtree push --prefix=python python-public main
   ```
3. This triggers the GitHub Actions workflow in the public repository
4. Documentation is built and deployed to GitHub Pages

Remember that paths in the public repository are different from the main repository:
- `python/docs/` in the main repo becomes `docs/` in the public repo
- `python/mkdocs.yml` in the main repo becomes `mkdocs.yml` in the public repo
- `python/scripts/` in the main repo becomes `scripts/` in the public repo

When editing documentation, always test locally before pushing to ensure everything works correctly in the public repository context.