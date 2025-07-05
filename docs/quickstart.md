# Quick Start Guide

Get up and running with DeepSentinel in 5 minutes. This guide shows you how to install the SDK and make your first compliant API call.

## System Requirements

Before installing DeepSentinel, ensure your system meets these requirements:

- **Python**: 3.8 or higher
- **Operating Systems**: Windows 10+, macOS 10.14+, Linux (any modern distribution)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for production)
- **Disk Space**: At least 100MB free space
- **Network**: Internet connection for API access
- **Dependencies**: pip package manager

## Installation Options

### Standard Installation

Install the DeepSentinel SDK using pip:

```bash
pip install deepsentinel-sdk```

### Development Installation

For development with additional tools and dependencies:

```bash
# Clone the repository
git clone https://github.com/deepsentinel/deepsentinel-sdk.git
cd deepsentinel-sdk/python

# Install in development mode with all extras
pip install -e ".[dev,test,docs]"
```

### Virtual Environment (Recommended)

We recommend using a virtual environment for isolation:

```bash
# Create virtual environment
python -m venv deepsentinel-env

# Activate on Windows
deepsentinel-env\Scripts\activate

# Activate on macOS/Linux
source deepsentinel-env/bin/activate

# Install DeepSentinel
pip install deepsentinel-sdk```

## Verify Installation

After installation, verify that DeepSentinel is correctly installed:

```bash
# Check the installed version
python -c "import deepsentinel; print(f'DeepSentinel SDK version: {deepsentinel.__version__}')"

# Run the verification script (if installed in development mode)
python -m deepsentinel.verify
```

## Get API Keys

You'll need:

1. **DeepSentinel API Key** - [Sign up for free](https://deepsentinel.ai/signup)
2. **LLM Provider API Key** - From [OpenAI](https://platform.openai.com/api-keys), [Anthropic](https://console.anthropic.com/), etc.

## Basic Usage

Create a new Python file and add this code:

```python
from deepsentinel import SentinelClient

# Initialize the client
client = SentinelClient(
    sentinel_api_key="your-deepsentinel-api-key",
    openai_api_key="your-openai-api-key"
)

# Make a chat completion request
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the key benefits of AI compliance?"}
    ]
)

# Print the response
print(response.choices[0].message.content)
```

## Run Your Code

Save the file as `test_deepsentinel.py` and run it:

```bash
python test_deepsentinel.py
```

That's it! 🎉 You've just made your first compliant AI request.

## What Just Happened?

Behind the scenes, DeepSentinel:

1. ✅ **Scanned your request** for sensitive data (PII, PHI, etc.)
2. ✅ **Applied compliance policies** based on your configuration
3. ✅ **Logged the interaction** for audit purposes
4. ✅ **Sent the request to OpenAI** using their API
5. ✅ **Returned the response** through the same interface

## Testing Compliance

Try sending a request with sensitive data to see DeepSentinel in action:

```python
from deepsentinel import SentinelClient, ComplianceViolationError

client = SentinelClient(
    sentinel_api_key="your-deepsentinel-api-key",
    openai_api_key="your-openai-api-key"
)

try:
    # This request contains a credit card number
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user", 
            "content": "My credit card number is 4111-1111-1111-1111"
        }]
    )
except ComplianceViolationError as e:
    print(f"Compliance violation detected: {e}")
    print(f"Violation types: {e.violations}")
```

DeepSentinel will block this request and raise a `ComplianceViolationError` because it detected a credit card number.

## Configuration Options

You can customize DeepSentinel's behavior during initialization:

```python
from deepsentinel import SentinelClient, CompliancePolicy

# Create a custom compliance policy
policy = CompliancePolicy(
    name="strict-policy",
    jurisdictions=["GDPR", "HIPAA", "CCPA"],
    block_on_violation=True,
    max_risk_score=0.7  # More strict than default (0.8)
)

# Initialize with custom configuration
client = SentinelClient(
    sentinel_api_key="your-deepsentinel-api-key",
    openai_api_key="your-openai-api-key",
    compliance_policies=[policy],
    debug_mode=True  # Enable debug logging
)
```

## Using Different Providers

DeepSentinel supports multiple LLM providers with the same interface:

```python
client = SentinelClient(
    sentinel_api_key="your-deepsentinel-api-key",
    openai_api_key="your-openai-api-key",
    anthropic_api_key="your-anthropic-api-key"
)

# Use OpenAI (default)
openai_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello from OpenAI!"}]
)

# Use Anthropic
anthropic_response = client.chat.completions.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "Hello from Anthropic!"}],
    provider="anthropic"  # Specify the provider
)
```

## Streaming Responses

DeepSentinel fully supports streaming responses:

```python
# Create a streaming request
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem about AI safety"}],
    stream=True
)

# Print the response as it arrives
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Environment Variables

For production applications, use environment variables for API keys:

```bash
# Set environment variables
export DEEPSENTINEL_API_KEY="your-deepsentinel-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

```python
import os
from deepsentinel import SentinelClient

# DeepSentinel automatically reads from environment variables
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
```

## Troubleshooting Installation

### Package Not Found
```
ERROR: Could not find a version that satisfies the requirement deepsentinel
ERROR: No matching distribution found for deepsentinel
```
**Solution**: 
- Check your internet connection
- Verify you're using Python 3.8+: `python --version`
- Ensure pip is up to date: `pip install --upgrade pip`

### Dependency Conflicts
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```
**Solution**: Use a virtual environment for a clean installation:
```bash
python -m venv deepsentinel-env
source deepsentinel-env/bin/activate  # Windows: deepsentinel-env\Scripts\activate
pip install deepsentinel-sdk```

### Permission Errors
```
ERROR: Could not install packages due to PermissionError
```
**Solution**: 
- Use `--user` flag: `pip install --user deepsentinel`
- On Unix systems, use `sudo` (not recommended): `sudo pip install deepsentinel-sdk`
- Use a virtual environment (recommended)

### Import Error
```
ImportError: No module named 'deepsentinel'
```
**Solution**: 
- Verify installation: `pip show deepsentinel`
- Check Python path: `python -c "import sys; print(sys.path)"`
- Install again with verbose output: `pip install -v deepsentinel`

### SSL Certificate Errors
```
URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
```
**Solution**:
- Update certificate authorities: `pip install --upgrade certifi`
- On macOS, run the Python Install Certificates command: `/Applications/Python X.Y/Install Certificates.command`

## Next Steps

Now that you have DeepSentinel working, explore more advanced features:

- **[Core Concepts](concepts/overview.md)** - Learn how DeepSentinel works under the hood
- **[Tutorials](tutorials/basic-integration.md)** - Step-by-step guides for common scenarios
- **[API Reference](reference/)** - Complete documentation of all classes and methods
- **[Best Practices](guides/best-practices.md)** - Guidelines for production deployments

## Common Issues

### Import Error
```
ImportError: No module named 'deepsentinel'
```
**Solution**: Make sure you've installed the package: `pip install deepsentinel-sdk`

### Authentication Error
```
AuthenticationError: Invalid API key
```
**Solution**: Check that your API keys are correct and have proper permissions

### Configuration Error
```
ConfigurationError: No providers configured
```
**Solution**: Make sure you've provided at least one LLM provider API key

## Getting Help

If you run into issues:

- Check the [FAQ](faq.md) for common questions
- Review the [API Reference](reference/) for detailed documentation
- Open an issue on [GitHub](https://github.com/deepsentinel/deepsentinel-sdk/issues)
- Contact support at [support@deepsentinel.ai](mailto:support@deepsentinel.ai)

---

Ready for more advanced usage? Continue to [Core Concepts →](concepts/overview.md)