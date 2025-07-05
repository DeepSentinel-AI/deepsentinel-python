# DeepSentinel SDK Developer Guide

**AI compliance middleware for safe LLM interactions**

This is the comprehensive developer guide for the DeepSentinel Python SDK. Use this as your primary reference for integrating AI compliance into your applications.

## Table of Contents

- [Getting Started](#getting-started)
- [Core Concepts](#core-concepts)
- [Detailed Usage Guides](#detailed-usage-guides)
- [Advanced Configuration](#advanced-configuration)
- [Real Examples](#real-examples)
- [Performance & Optimization](#performance--optimization)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

Install the DeepSentinel SDK from PyPI:

```bash
pip install deepsentinel-sdk```

### Quick Start - 5 Minutes to Compliance

Replace your existing OpenAI or Anthropic calls with DeepSentinel:

```python
from deepsentinel import SentinelClient

# Initialize the client (replaces OpenAI() or Anthropic())
client = SentinelClient(
    sentinel_api_key="your-deepsentinel-api-key",
    openai_api_key="your-openai-api-key"
)

# Use exactly like OpenAI's SDK - same interface, same parameters
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are best practices for data privacy?"}
    ]
)

print(response.choices[0].message.content)
```

**That's it!** DeepSentinel now automatically:
- ✅ Scans requests for PII, PHI, and sensitive data
- ✅ Applies compliance policies
- ✅ Logs all interactions for audit
- ✅ Routes to your chosen LLM provider

### Environment Variables

For production, use environment variables:

```bash
export DEEPSENTINEL_API_KEY="your-deepsentinel-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

```python
# DeepSentinel automatically reads environment variables
client = SentinelClient()
```

## Core Concepts

### What DeepSentinel Does

DeepSentinel sits as an intelligent middleware layer between your application and LLM providers:

```
Your App → DeepSentinel SDK → Compliance Engine → LLM Provider → Response
```

**Request Flow:**
1. **Pre-Processing**: Fast local detection of obvious sensitive patterns
2. **Compliance Analysis**: Deep scanning for PII, PHI, PCI violations
3. **Policy Enforcement**: Block, allow, or anonymize based on your rules
4. **Provider Routing**: Send approved requests to OpenAI, Anthropic, etc.
5. **Response Processing**: Final compliance check on responses
6. **Audit Logging**: Complete activity logging for compliance

### Key Components

#### SentinelClient
Your main interface - drop-in replacement for provider SDKs:

```python
from deepsentinel import SentinelClient

client = SentinelClient(
    sentinel_api_key="your-key",
    openai_api_key="your-openai-key",
    anthropic_api_key="your-anthropic-key"  # Optional
)

# Familiar interfaces
client.chat.completions.create(...)      # Chat completions
client.embeddings.create(...)            # Embeddings
client.models.list()                     # Model listing
client.images.generate(...)              # Image generation
```

#### Provider System
Unified interface across multiple LLM providers:

```python
# OpenAI (default)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Anthropic (explicit)
response = client.chat.completions.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "Hello!"}],
    provider="anthropic"
)
```

#### Compliance Engine
Automatic detection and policy enforcement:

- **PII Detection**: Emails, phones, SSNs, addresses, names
- **PHI Detection**: Medical records, prescriptions, health conditions
- **PCI Detection**: Credit cards, bank accounts, payment data
- **Custom Patterns**: Your organization-specific sensitive data

## Detailed Usage Guides

### Working with Providers

#### Multiple Provider Setup

```python
from deepsentinel import SentinelClient

# Configure multiple providers
client = SentinelClient(
    sentinel_api_key="your-key",
    openai_api_key="your-openai-key",
    anthropic_api_key="your-anthropic-key"
)

# Use different providers for different tasks
def get_response(prompt, task_type):
    if task_type == "analysis":
        # Use Claude for deep analysis
        return client.chat.completions.create(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": prompt}],
            provider="anthropic"
        )
    elif task_type == "code":
        # Use GPT-4 for code generation
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            provider="openai"
        )
```

#### Provider Registry Advanced Usage

```python
from deepsentinel.providers.registry import get_global_registry

# Get registry for advanced operations
registry = get_global_registry()

# Check provider health
health = await registry.health_check_all()
for provider, status in health.items():
    print(f"{provider}: {status['status']}")

# Get models for specific provider
models = await registry.get_provider("openai").list_models()
print(f"Available models: {[m['id'] for m in models]}")
```

### Compliance Policies

#### Basic Compliance Setup

```python
from deepsentinel import SentinelClient, CompliancePolicy

# Create a compliance policy
policy = CompliancePolicy(
    name="basic-compliance",
    jurisdictions=["GDPR", "HIPAA", "CCPA"],
    block_on_violation=True,
    max_risk_score=0.8,
    enable_pii_detection=True,
    enable_phi_detection=True,
    enable_pci_detection=True
)

client = SentinelClient(
    sentinel_api_key="your-key",
    openai_api_key="your-openai-key",
    compliance_policies=[policy]
)
```

#### Specialized Policies

```python
from deepsentinel import PIIPolicy, ContentFilterPolicy

# PII-specific policy
pii_policy = PIIPolicy(
    name="strict-pii",
    detection_threshold=0.8,
    redaction_strategy="mask",
    pii_types=["email", "phone", "ssn", "credit_card"],
    allow_partial_redaction=True
)

# Content filtering policy
content_policy = ContentFilterPolicy(
    name="content-safety",
    filter_categories=["hate", "violence", "harassment"],
    strictness_level="medium",
    context_aware=True
)

# Apply multiple policies
client = SentinelClient(
    sentinel_api_key="your-key",
    openai_api_key="your-openai-key",
    compliance_policies=[pii_policy, content_policy]
)
```

#### Handling Violations

```python
from deepsentinel import ComplianceViolationError

def safe_chat_completion(messages):
    try:
        return client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
    except ComplianceViolationError as e:
        print(f"Compliance violation: {e.message}")
        print(f"Policy: {e.policy_name}")
        print(f"Violation type: {e.violation_type}")
        print(f"Severity: {e.severity}")
        
        # Handle based on violation type
        if e.violation_type == "pii":
            return {"error": "Request contains personal information"}
        elif e.violation_type == "content_filter":
            return {"error": "Request contains inappropriate content"}
        else:
            return {"error": "Request violates compliance policy"}

# Test with sensitive data
response = safe_chat_completion([
    {"role": "user", "content": "My credit card is 4111-1111-1111-1111"}
])
```

### Streaming Responses

DeepSentinel fully supports streaming with compliance:

```python
def stream_with_compliance(prompt):
    """Stream responses while maintaining compliance."""
    try:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        response_text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                response_text += content
        
        return response_text
        
    except ComplianceViolationError as e:
        print(f"\n[BLOCKED] {e.message}")
        return None

# Use streaming
response = stream_with_compliance("Write a story about data privacy")
```

### Error Handling

#### Comprehensive Error Handling

```python
from deepsentinel import (
    DeepSentinelError,
    ComplianceViolationError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ConfigurationError,
    ValidationError
)

def robust_chat_completion(messages, model="gpt-4o"):
    """Robust chat completion with comprehensive error handling."""
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages
        )
        
    except ComplianceViolationError as e:
        # Handle compliance violations
        print(f"🚫 Compliance violation: {e.message}")
        return {"error": "compliance", "details": e.details}
        
    except AuthenticationError as e:
        # Handle authentication issues
        print(f"🔐 Authentication failed for {e.provider_name}: {e.message}")
        return {"error": "auth", "provider": e.provider_name}
        
    except RateLimitError as e:
        # Handle rate limiting
        print(f"⏱️ Rate limit exceeded for {e.provider_name}")
        if e.retry_after:
            print(f"Retry after {e.retry_after} seconds")
        return {"error": "rate_limit", "retry_after": e.retry_after}
        
    except ProviderError as e:
        # Handle provider-specific errors
        print(f"🔧 Provider error ({e.provider_name}): {e.message}")
        return {"error": "provider", "provider": e.provider_name}
        
    except ValidationError as e:
        # Handle validation errors
        print(f"❌ Validation error: {e.message}")
        if e.field_name:
            print(f"Field: {e.field_name} = {e.field_value}")
        return {"error": "validation", "field": e.field_name}
        
    except ConfigurationError as e:
        # Handle configuration errors
        print(f"⚙️ Configuration error: {e.message}")
        return {"error": "config", "key": e.config_key}
        
    except DeepSentinelError as e:
        # Handle any other SDK errors
        print(f"🛡️ DeepSentinel error: {e.message}")
        return {"error": "sdk", "code": e.error_code}
        
    except Exception as e:
        # Handle unexpected errors
        print(f"💥 Unexpected error: {e}")
        return {"error": "unexpected", "message": str(e)}

# Usage
result = robust_chat_completion([
    {"role": "user", "content": "Hello, world!"}
])
```

## Advanced Configuration

### Complete Configuration Example

```python
from deepsentinel import (
    SentinelConfig,
    CompliancePolicy,
    AuditConfig,
    LoggingConfig,
    PerformanceConfig,
    ProviderConfig
)

# Comprehensive configuration
config = SentinelConfig(
    # Basic settings
    debug_mode=False,
    environment="production",
    timeout=30,
    max_retries=3,
    
    # Provider configurations
    providers={
        "openai": ProviderConfig(
            provider_type="openai",
            api_key="your-openai-key",
            base_url="https://api.openai.com/v1",
            timeout=30,
            max_retries=3
        ),
        "anthropic": ProviderConfig(
            provider_type="anthropic",
            api_key="your-anthropic-key",
            base_url="https://api.anthropic.com/v1",
            timeout=45,
            max_retries=2
        )
    },
    default_provider="openai",
    
    # Compliance policies
    compliance_policies=[
        CompliancePolicy(
            name="production-compliance",
            jurisdictions=["GDPR", "HIPAA", "CCPA"],
            block_on_violation=True,
            max_risk_score=0.7,
            enable_pii_detection=True,
            enable_phi_detection=True,
            enable_pci_detection=True
        )
    ],
    
    # Audit configuration
    audit_config=AuditConfig(
        enabled=True,
        log_level="INFO",
        storage_backend="file",
        retention_days=90,
        include_request_body=True,
        include_response_body=False,  # Privacy-conscious
        encryption_enabled=True
    ),
    
    # Logging configuration
    logging_config=LoggingConfig(
        level="INFO",
        structured=True,
        include_timestamp=True,
        include_trace_id=True
    ),
    
    # Performance configuration
    performance_config=PerformanceConfig(
        enable_caching=True,
        cache_ttl=3600,
        cache_max_size=1000,
        connection_pool_size=100,
        connection_pool_per_host=20,
        enable_metrics=True,
        metrics_window_size=1000
    )
)

# Initialize client with comprehensive config
client = SentinelClient(config=config)
```

### Configuration from File

```python
# deepsentinel.yaml
sentinel_api_key: "your-deepsentinel-key"
debug_mode: false
environment: "production"

providers:
  openai:
    provider_type: "openai"
    api_key: "your-openai-key"
    timeout: 30
  anthropic:
    provider_type: "anthropic"
    api_key: "your-anthropic-key"
    timeout: 45

compliance_policies:
  - name: "production-policy"
    jurisdictions: ["GDPR", "HIPAA"]
    block_on_violation: true
    max_risk_score: 0.8

performance_config:
  enable_caching: true
  cache_ttl: 3600
  enable_metrics: true
```

```python
# Load from file
config = SentinelConfig.from_file("deepsentinel.yaml")
client = SentinelClient(config=config)
```

## Real Examples

### Example 1: Customer Support Chatbot

```python
from deepsentinel import SentinelClient, PIIPolicy

class CustomerSupportBot:
    def __init__(self):
        # Configure strict PII protection for customer data
        pii_policy = PIIPolicy(
            name="customer-pii-protection",
            detection_threshold=0.7,
            redaction_strategy="mask",
            pii_types=["email", "phone", "ssn", "address", "name"],
            block_on_violation=False,  # Allow but mask
            allow_partial_redaction=True
        )
        
        self.client = SentinelClient(
            sentinel_api_key="your-key",
            openai_api_key="your-openai-key",
            compliance_policies=[pii_policy]
        )
    
    def handle_customer_query(self, query, context=None):
        """Handle customer support queries with PII protection."""
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful customer support agent. Do not repeat any personal information like emails, phone numbers, or addresses."
            }
        ]
        
        if context:
            messages.append({
                "role": "assistant",
                "content": f"I see you're asking about: {context}"
            })
        
        messages.append({"role": "user", "content": query})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3  # Conservative for support
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return "I apologize, but I cannot process your request right now. Please contact support directly."

# Usage
bot = CustomerSupportBot()
response = bot.handle_customer_query(
    "I need help with my account. My email is john@example.com and phone is 555-123-4567"
)
print(response)  # PII will be masked automatically
```

### Example 2: Healthcare Application

```python
from deepsentinel import SentinelClient, CompliancePolicy

class HealthcareAssistant:
    def __init__(self):
        # HIPAA-compliant configuration
        hipaa_policy = CompliancePolicy(
            name="hipaa-compliance",
            jurisdictions=["HIPAA"],
            block_on_violation=True,  # Strict blocking for healthcare
            max_risk_score=0.6,      # Very conservative
            enable_phi_detection=True,
            enable_pii_detection=True
        )
        
        self.client = SentinelClient(
            sentinel_api_key="your-key",
            openai_api_key="your-openai-key",
            compliance_policies=[hipaa_policy]
        )
    
    def get_health_information(self, query):
        """Provide health information while ensuring HIPAA compliance."""
        messages = [
            {
                "role": "system",
                "content": "You are a healthcare information assistant. Provide general health information only. Do not diagnose, prescribe, or provide personalized medical advice."
            },
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1  # Very conservative for healthcare
            )
            return response.choices[0].message.content
            
        except ComplianceViolationError as e:
            return "I cannot process requests containing personal health information. Please ask for general health information only."
        except Exception as e:
            return "I apologize, but I cannot provide health information at this time."

# Usage
assistant = HealthcareAssistant()
response = assistant.get_health_information(
    "What are the general symptoms of diabetes?"  # Safe query
)
```

### Example 3: Financial Services

```python
from deepsentinel import SentinelClient, CompliancePolicy

class FinancialAdvisor:
    def __init__(self):
        # Financial compliance (PCI + general)
        financial_policy = CompliancePolicy(
            name="financial-compliance",
            jurisdictions=["PCI", "GDPR"],
            block_on_violation=True,
            max_risk_score=0.7,
            enable_pci_detection=True,
            enable_pii_detection=True
        )
        
        self.client = SentinelClient(
            sentinel_api_key="your-key",
            openai_api_key="your-openai-key",
            compliance_policies=[financial_policy]
        )
    
    def get_financial_advice(self, query):
        """Provide financial advice while protecting sensitive financial data."""
        messages = [
            {
                "role": "system",
                "content": "You are a financial advisor. Provide general financial guidance. Never request or reference specific account numbers, credit card numbers, or personal financial details."
            },
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2
            )
            return response.choices[0].message.content
            
        except ComplianceViolationError as e:
            return "I cannot process requests containing sensitive financial information. Please remove any account numbers, credit card numbers, or personal financial details."

# Usage
advisor = FinancialAdvisor()
response = advisor.get_financial_advice(
    "What are some general strategies for saving for retirement?"
)
```

### Example 4: Migration from OpenAI

Before (Direct OpenAI):
```python
import openai

openai.api_key = "your-openai-key"

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

After (With DeepSentinel):
```python
from deepsentinel import SentinelClient

client = SentinelClient(
    sentinel_api_key="your-deepsentinel-key",
    openai_api_key="your-openai-key"
)

# Same interface, added compliance
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Performance & Optimization

### Performance Configuration

```python
from deepsentinel import SentinelClient, PerformanceConfig

# Optimize for high-throughput applications
perf_config = PerformanceConfig(
    # Caching
    enable_caching=True,
    cache_ttl=3600,        # 1 hour cache
    cache_max_size=5000,   # Large cache for high volume
    
    # Connection pooling
    connection_pool_size=200,     # High concurrency
    connection_pool_per_host=50,  # Many connections per provider
    connect_timeout=10,
    read_timeout=60,
    
    # Metrics
    enable_metrics=True,
    metrics_window_size=10000,    # Large metrics window
    
    # Pattern matching optimization
    pattern_cache_size=2000       # Cache compiled patterns
)

client = SentinelClient(
    sentinel_api_key="your-key",
    openai_api_key="your-openai-key",
    performance_config=perf_config
)
```

### Monitoring Performance

```python
from deepsentinel import MetricsCollector

# Access metrics collector
metrics = client.metrics_collector

# Get overall performance metrics
overall = metrics.get_overall_metrics()
print(f"Total requests: {overall['performance']['request_count']}")
print(f"Success rate: {overall['performance']['success_rate']:.2%}")
print(f"Average latency: {overall['performance']['avg_duration']:.3f}s")
print(f"Cache hit rate: {overall['cache']['hit_rate']:.2%}")

# Get provider-specific metrics
provider_metrics = metrics.get_provider_metrics()
for provider, data in provider_metrics.items():
    print(f"{provider}:")
    print(f"  Requests: {data['performance']['request_count']}")
    print(f"  Success rate: {data['performance']['success_rate']:.2%}")
    print(f"  Avg latency: {data['performance']['avg_duration']:.3f}s")
```

### Health Monitoring

```python
async def monitor_health():
    """Monitor system health and provider status."""
    health = await client.health_check()
    
    print(f"Overall status: {health['status']}")
    
    # Check provider health
    for provider, status in health['providers'].items():
        print(f"{provider}: {status['status']}")
        if status['status'] != 'healthy':
            print(f"  Error: {status.get('error', 'Unknown')}")
    
    # Check middleware health
    middleware = health['middleware']
    print(f"Middleware: {middleware['status']}")
    print(f"Active policies: {middleware['policies']}")

# Run health check
import asyncio
asyncio.run(monitor_health())
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Installation Issues
```bash
# Problem: Package not found
pip install deepsentinel-sdk
# Problem: Dependencies conflict
pip install --upgrade deepsentinel

# Problem: Import errors
python -c "import deepsentinel; print('OK')"
```

#### 2. Authentication Issues
```python
# Problem: Invalid API key
try:
    client = SentinelClient(
        sentinel_api_key="invalid-key",
        openai_api_key="your-openai-key"
    )
    response = client.chat.completions.create(...)
except AuthenticationError as e:
    print(f"Auth failed for {e.provider_name}: {e.message}")
    # Check your API keys and permissions
```

#### 3. Configuration Issues
```python
# Problem: No providers configured
try:
    client = SentinelClient(sentinel_api_key="your-key")
    # Missing provider API key
except ConfigurationError as e:
    print(f"Config error: {e.message}")
    print(f"Problem with: {e.config_key}")
```

#### 4. Compliance Issues
```python
# Problem: Unexpected compliance violations
try:
    response = client.chat.completions.create(...)
except ComplianceViolationError as e:
    print(f"Violation: {e.violation_type}")
    print(f"Policy: {e.policy_name}")
    print(f"Details: {e.details}")
    
    # Adjust your compliance policy if needed
    # Or sanitize your input data
```

#### 5. Performance Issues
```python
# Problem: Slow response times
import time

start_time = time.time()
response = client.chat.completions.create(...)
duration = time.time() - start_time

print(f"Request took {duration:.2f} seconds")

# Solutions:
# 1. Enable caching
# 2. Increase connection pool size
# 3. Use local detection for simple patterns
# 4. Optimize compliance policies
```

### Debug Mode

```python
# Enable debug mode for troubleshooting
client = SentinelClient(
    sentinel_api_key="your-key",
    openai_api_key="your-openai-key",
    debug_mode=True  # Enables detailed logging
)

# Or set environment variable
# export DEEPSENTINEL_DEBUG=true
```

### Logging Configuration

```python
import logging
from deepsentinel import LoggingConfig

# Configure detailed logging
logging_config = LoggingConfig(
    level="DEBUG",
    structured=True,
    include_timestamp=True,
    include_trace_id=True
)

client = SentinelClient(
    sentinel_api_key="your-key",
    openai_api_key="your-openai-key",
    logging_config=logging_config
)
```

### Getting Help

When you need assistance:

1. **Check the Logs**: Enable debug mode and check logs for detailed error information
2. **Review Configuration**: Verify your API keys, policies, and provider settings
3. **Test Incrementally**: Start with basic requests and add complexity gradually
4. **Check Provider Status**: Verify that your LLM providers (OpenAI, Anthropic) are operational
5. **Update SDK**: Ensure you're using the latest version: `pip install --upgrade deepsentinel`

**Support Resources:**
- 📖 **API Documentation**: [Complete API reference](https://deepsentinel-ai.github.io/deepsentinel-python)
- 💬 **Community**: [GitHub Issues](https://github.com/deepsentinel/deepsentinel-sdk/issues)
- 📧 **Support**: [support@deepsentinel.ai](mailto:support@deepsentinel.ai)
- 🔗 **Examples**: [GitHub Examples](https://github.com/deepsentinel/deepsentinel-sdk/tree/main/examples/python)

---

## Summary

DeepSentinel SDK provides enterprise-grade AI compliance with a familiar interface. Key benefits:

- **🛡️ Automatic Compliance**: PII, PHI, PCI detection and policy enforcement
- **🔄 Multi-Provider**: Unified interface for OpenAI, Anthropic, and more
- **⚡ High Performance**: Optimized with caching, connection pooling, and metrics
- **📊 Full Observability**: Comprehensive logging, monitoring, and audit trails
- **🔧 Easy Integration**: Drop-in replacement for existing LLM SDK calls

Start with the basic examples above, then customize policies and configuration for your specific compliance requirements.

**Happy coding with compliant AI! 🚀**