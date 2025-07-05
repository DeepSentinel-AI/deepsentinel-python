# DeepSentinel SDK - PyPI Packaging Guide

This guide documents the PyPI packaging configuration and release process for the DeepSentinel SDK Python package.

## Package Configuration

### Core Metadata
- **Package Name**: `deepsentinel`
- **Current Version**: `0.1.0` (managed in `src/deepsentinel/__init__.py`)
- **Description**: Developer-friendly SDK providing AI compliance middleware for safe LLM interactions
- **License**: MIT
- **Python Support**: 3.8+
- **Build System**: Hatchling

### Dependencies

#### Core Runtime Dependencies
- `httpx>=0.24.0` - Modern HTTP client
- `pydantic>=2.0.0` - Data validation and serialization
- `typing-extensions>=4.0.0` - Type hints support
- `openai>=1.0.0` - OpenAI API client
- `anthropic>=0.25.0` - Anthropic API client
- `aiohttp>=3.8.0` - Async HTTP client
- `asyncio-mqtt>=0.13.0` - MQTT async client
- `tenacity>=8.0.0` - Retry library
- `structlog>=23.0.0` - Structured logging
- `click>=8.0.0` - CLI framework

#### Optional Dependencies

**Development (`deepsentinel[dev]`)**:
- pytest, pytest-asyncio, pytest-cov, pytest-mock
- black, isort, flake8, mypy
- pre-commit, twine, build

**Documentation (`deepsentinel[docs]`)**:
- mkdocs, mkdocs-material
- mkdocstrings, mkdocs-gen-files
- mkdocs-literate-nav, mkdocs-section-index

**Testing (`deepsentinel[test]`)**:
- pytest suite with async support
- HTTP mocking libraries (responses, httpx-mock)

## Package Structure

```
python/
├── src/deepsentinel/           # Main package source
│   ├── __init__.py            # Version and exports
│   ├── client.py              # Main client
│   ├── config.py              # Configuration
│   ├── exceptions.py          # Custom exceptions
│   ├── types.py               # Type definitions
│   ├── api/                   # API clients
│   ├── audit/                 # Audit functionality  
│   ├── cache/                 # Caching layer
│   ├── compliance/            # Compliance engine
│   ├── metrics/               # Performance metrics
│   ├── middleware/            # Middleware components
│   └── providers/             # LLM provider integrations
├── tests/                     # Test suite
├── docs/                      # Documentation
├── scripts/                   # Build and utility scripts
├── pyproject.toml            # Package configuration
├── README.md                 # Package documentation
└── LICENSE                   # MIT license
```

## Build Scripts

### `scripts/build-package.py`
Comprehensive build script that:
- Verifies package structure
- Cleans previous build artifacts
- Builds both wheel and source distributions
- Validates build output

### `scripts/test-installation.py`
Installation verification script that:
- Creates isolated test environment
- Installs the built package
- Tests basic imports and functionality
- Validates core components work correctly

### `scripts/prepare-release.py`
End-to-end release preparation that:
- Checks version consistency
- Runs build process
- Executes installation tests
- Provides next steps for PyPI publication

## Usage

### Installing the Package
```bash
# Install from PyPI (when published)
pip install deepsentinel-sdk
# Install with optional dependencies
pip install deepsentinel[dev]    # Development dependencies
pip install deepsentinel[docs]   # Documentation dependencies  
pip install deepsentinel[test]   # Testing dependencies
```

### Basic Usage
```python
import deepsentinel
from deepsentinel import SentinelClient, SentinelConfig

# Create configuration
config = SentinelConfig(
    api_key="your-api-key",
    enable_compliance=True
)

# Initialize client
client = SentinelClient(config)

# Use the client for AI interactions
```

## Release Process

### 1. Prepare Release
```bash
cd python
python scripts/prepare-release.py
```

### 2. Test on TestPyPI (Recommended)
```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ deepsentinel

# Verify installation works
python -c "import deepsentinel; print(deepsentinel.__version__)"
```

### 3. Upload to PyPI
```bash
# Upload to production PyPI
python -m twine upload dist/*
```

### 4. Verify Production Release
```bash
# Install from PyPI
pip install deepsentinel-sdk
# Test basic functionality
python -c "
from deepsentinel import SentinelClient, SentinelConfig
config = SentinelConfig()
print('✅ DeepSentinel SDK installation verified')
"
```

## Version Management

Version is managed in `src/deepsentinel/__init__.py`:
```python
__version__ = "0.1.0"
```

Hatchling automatically reads this version for package metadata.

### Version Update Process
1. Update `__version__` in `src/deepsentinel/__init__.py`
2. Run preparation script to verify consistency
3. Build and test the package
4. Create git tag: `git tag v0.1.0`
5. Push tag: `git push origin v0.1.0`
6. Release to PyPI

## Authentication Setup

### Option 1: API Token (Recommended)
Create `~/.pypirc`:
```ini
[distutils]
index-servers = pypi testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-testpypi-token-here
```

### Option 2: Environment Variables
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-api-token-here
```

## Quality Assurance

### Pre-release Checklist
- [ ] Version updated in `__init__.py`
- [ ] All tests pass: `pytest`
- [ ] Code formatted: `black src/ tests/`
- [ ] Imports sorted: `isort src/ tests/`
- [ ] Type checking: `mypy src/`
- [ ] Build script passes: `python scripts/prepare-release.py`
- [ ] Installation test passes in clean environment
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

### Continuous Integration
The package build and test scripts can be integrated into CI/CD pipelines:
```yaml
# Example GitHub Actions workflow
- name: Build Package
  run: cd python && python scripts/build-package.py

- name: Test Installation  
  run: cd python && python scripts/test-installation.py
```

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are correctly specified in `pyproject.toml`

**Build Failures**: Check that all required files are present and `pyproject.toml` is valid

**Upload Errors**: Verify authentication credentials and package name availability

**Version Conflicts**: Ensure version in `__init__.py` follows semantic versioning

### Support
For packaging issues, check:
1. This packaging guide
2. Build script output for specific error messages
3. PyPI documentation: https://packaging.python.org/
4. Hatchling documentation: https://hatch.pylib.io/