# Frequently Asked Questions

Common questions and solutions for using the DeepSentinel Python SDK.

## Getting Started

### Q: How do I get a DeepSentinel API key?

**A:** Sign up for a free account at [deepsentinel.ai/signup](https://deepsentinel.ai/signup). You'll receive an API key immediately after registration.

### Q: Do I need to change my existing OpenAI code?

**A:** Minimal changes are required. DeepSentinel provides the same interface as OpenAI's SDK:

```python
# Before
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# After  
from deepsentinel import SentinelClient
client = SentinelClient(
    sentinel_api_key="ds-...",
    openai_api_key="sk-..."
)

# Rest of your code stays the same!
response = client.chat.completions.create(...)
```

### Q: What Python versions are supported?

**A:** DeepSentinel supports Python 3.8 and higher. We recommend using Python 3.9+ for the best experience.

## Installation and Setup

### Q: How do I install DeepSentinel?

**A:** Install using pip:

```bash
pip install deepsentinel-sdk```

For development with all optional dependencies:

```bash
pip install "deepsentinel[dev]"
```

### Q: I'm getting an import error. What should I do?

**A:** Make sure DeepSentinel is properly installed:

```bash
pip show deepsentinel
```

If not installed, run:

```bash
pip install deepsentinel-sdk```

If you're still having issues, try:
1. Ensure you're using a supported Python version
2. Try installing in a clean virtual environment
3. Update pip: `pip install --upgrade pip`
4. Check your PYTHONPATH environment variable

### Q: Can I use DeepSentinel with virtual environments?

**A:** Yes! We recommend using virtual environments:

```bash
python -m venv deepsentinel-env
source deepsentinel-env/bin/activate  # On Windows: deepsentinel-env\Scripts\activate
pip install deepsentinel-sdk```

### Q: Is DeepSentinel compatible with containerized environments like Docker?

**A:** Yes, DeepSentinel works well in containerized environments. Add it to your Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install deepsentinel-sdk
# Set environment variables
ENV DEEPSENTINEL_API_KEY="your-key"
ENV OPENAI_API_KEY="your-openai-key"

COPY . .
CMD ["python", "your_app.py"]
```

## Configuration

### Q: How do I configure environment variables?

**A:** Set these environment variables:

```bash
export DEEPSENTINEL_API_KEY="your-deepsentinel-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional
```

Then initialize without explicit keys:

```python
client = SentinelClient()  # Automatically reads environment variables
```

### Q: Can I use configuration files?

**A:** Yes! Create a YAML configuration file:

```yaml
# deepsentinel.yaml
sentinel_api_key: "your-key"
default_provider: "openai"
providers:
  openai:
    api_key: "your-openai-key"
```

Load it in your code:

```python
from deepsentinel import SentinelConfig, SentinelClient

config = SentinelConfig.from_file("deepsentinel.yaml")
client = SentinelClient(config=config)
```

### Q: How do I adjust compliance strictness?

**A:** Configure the `max_risk_score` parameter:

```python
from deepsentinel import CompliancePolicy

# More strict (blocks more content)
strict_policy = CompliancePolicy(
    name="strict",
    max_risk_score=0.6  # Lower = more strict
)

# More lenient (allows more content)
lenient_policy = CompliancePolicy(
    name="lenient", 
    max_risk_score=0.9  # Higher = more lenient
)
```

### Q: What are the best practices for configuration?

**A:** Follow these best practices:

1. **Use environment variables for secrets** in production environments
2. **Create different policies** for development vs. production
3. **Start strict and loosen gradually** based on user feedback
4. **Enable detailed logging** during initial deployment
5. **Store configurations in version control** (without API keys)
6. **Create environment-specific configs** (dev, staging, production)

## Compliance and Security

### Q: What types of sensitive data does DeepSentinel detect?

**A:** DeepSentinel detects various types of sensitive information:

- **PII**: Email addresses, phone numbers, SSNs, addresses, names
- **PHI**: Medical record numbers, insurance IDs, health conditions
- **PCI**: Credit card numbers, bank accounts, CVV codes
- **Custom**: Organization-specific patterns you define

### Q: Why was my request blocked?

**A:** Your request likely contained sensitive data. Check the error details:

```python
from deepsentinel import ComplianceViolationError

try:
    response = client.chat.completions.create(...)
except ComplianceViolationError as e:
    print(f"Blocked because: {e.message}")
    print(f"Detected: {[v.type for v in e.violations]}")
    print(f"Risk score: {e.risk_score}")
```

### Q: Can I allowlist certain types of data?

**A:** Yes, create a custom policy:

```python
policy = CompliancePolicy(
    name="custom-policy",
    enable_pii_detection=True,
    enable_phi_detection=False,  # Disable PHI detection
    custom_allowlist=["email_addresses"]  # Allow emails
)
```

### Q: How do I handle false positives?

**A:** You can:

1. **Adjust risk thresholds**:
```python
policy = CompliancePolicy(max_risk_score=0.9)  # More lenient
```

2. **Use anonymization instead of blocking**:
```python
policy = CompliancePolicy(
    block_on_violation=False,
    anonymize_sensitive_data=True
)
```

3. **Create custom patterns** for your specific use case

### Q: Is my data sent to DeepSentinel's servers?

**A:** DeepSentinel uses a layered approach:

- **Local detection** runs on your machine (fast, private)
- **Cloud analysis** is used only for ambiguous cases
- You can configure `local_only=True` to disable cloud analysis entirely

### Q: How secure is DeepSentinel?

**A:** DeepSentinel employs multiple security measures:

1. **End-to-end encryption** for all API communication
2. **No storage of content** - data is analyzed in memory and not persisted
3. **Zero-trust architecture** for backend systems
4. **Regular security audits** and penetration testing
5. **SOC 2 Type II compliant** infrastructure
6. **Role-based access control** for audit logs

### Q: Are there scenarios where compliance might be bypassed?

**A:** Several scenarios to be aware of:

1. **Risk threshold too high**: If `max_risk_score` is set very high, sensitive data might pass through
2. **Detection disabled**: If you disable specific detectors (`enable_pii_detection=False`)
3. **Data obfuscation**: Intentionally altered sensitive data (e.g., adding spaces between digits)
4. **Custom content**: Novel formats of sensitive data not in standard patterns
5. **Specific allowlists**: If you explicitly allowlist certain types of data

To prevent accidental bypasses:
- Start with stricter settings
- Review audit logs regularly
- Use anonymization as a fallback
- Implement layered compliance (client and server-side)
- Update patterns regularly

### Q: How can I maintain GDPR compliance?

**A:** To ensure GDPR compliance:

1. **Enable anonymization**: 
```python
policy = CompliancePolicy(
    anonymize_sensitive_data=True, 
    jurisdictions=["GDPR"]
)
```

2. **Set appropriate data retention**:
```python
config = SentinelConfig(
    audit_log_retention_days=30  # Set appropriate retention period
)
```

3. **Implement right to be forgotten** using the audit client:
```python
client.audit.delete_user_data(user_id="user123")
```

4. **Use geographic restrictions** to comply with data transfer regulations:
```python
policy = CompliancePolicy(
    geo_restrictions=["EU_ONLY"]
)
```

## Provider Integration

### Q: Which LLM providers are supported?

**A:** Currently supported providers:

- ✅ **OpenAI** (GPT-4, GPT-3.5, embeddings, images)
- ✅ **Anthropic** (Claude 3 models)
- 🔄 **More providers coming soon**

### Q: Can I use multiple providers?

**A:** Yes! Configure multiple providers and choose per request:

```python
client = SentinelClient(
    sentinel_api_key="ds-...",
    openai_api_key="sk-...",
    anthropic_api_key="ant-..."
)

# Use OpenAI
openai_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    provider="openai"
)

# Use Anthropic
anthropic_response = client.chat.completions.create(
    model="claude-3-opus-20240229",
    messages=[...],
    provider="anthropic"
)
```

### Q: What happens if a provider is down?

**A:** Configure automatic failover:

```python
config = SentinelConfig(
    fallback_strategy={
        "primary": "openai",
        "fallbacks": ["anthropic"],
        "auto_failover": True
    }
)
```

### Q: Can I dynamically switch providers based on content?

**A:** Yes, you can implement content-based provider selection:

```python
def select_provider(content):
    """Select provider based on content complexity."""
    if len(content) > 1000 or "complex analysis" in content.lower():
        return "anthropic"  # Use Claude for complex tasks
    elif "image" in content.lower():
        return "openai"     # Use OpenAI for image-related tasks
    else:
        return "openai"     # Default to OpenAI
        
# Usage
user_content = "Please analyze this complex data..."
selected_provider = select_provider(user_content)

response = client.chat.completions.create(
    model="gpt-4o" if selected_provider == "openai" else "claude-3-opus-20240229",
    messages=[{"role": "user", "content": user_content}],
    provider=selected_provider
)
```

## Performance

### Q: How much latency does DeepSentinel add?

**A:** Performance impact is minimal:

- **Local detection**: < 5ms
- **Cloud analysis**: 50-200ms (only for ambiguous cases)
- **Caching**: Near-zero latency for repeated patterns

### Q: How do I optimize performance?

**A:** Several strategies:

1. **Enable caching**:
```python
config = SentinelConfig(
    cache_enabled=True,
    cache_ttl=3600  # 1 hour
)
```

2. **Use local-only detection**:
```python
config = SentinelConfig(local_detection_only=True)
```

3. **Tune risk thresholds** to reduce cloud analysis

4. **Use connection pooling** for high-volume applications:
```python
from deepsentinel import PerformanceConfig

perf_config = PerformanceConfig(
    connection_pool_size=50,  # Increase for high throughput
    connection_pool_per_host=10
)

config = SentinelConfig(performance_config=perf_config)
```

5. **Implement batch processing** for multiple requests:
```python
# Process multiple messages in one batch
results = await client.process_batch([
    {"messages": message_set_1, "model": "gpt-4o"},
    {"messages": message_set_2, "model": "gpt-4o"}
])
```

### Q: Can I monitor performance metrics?

**A:** Yes, DeepSentinel provides comprehensive metrics:

```python
metrics = client.metrics_collector
print(f"Average latency: {metrics.average_latency}ms")
print(f"Cache hit rate: {metrics.cache_hit_rate}%")
print(f"Compliance violations: {metrics.compliance_violations}")
```

For detailed analysis:

```python
# Get comprehensive metrics report
report = metrics.generate_report()

# Provider-specific performance
provider_metrics = metrics.get_provider_metrics()
for provider, stats in provider_metrics.items():
    print(f"{provider} avg. latency: {stats['avg_latency']}ms")
```

### Q: How can I improve DeepSentinel's performance in production?

**A:** For production optimization:

1. **Implement request batching** to reduce overhead
2. **Use async methods** for non-blocking operations
3. **Optimize compliance policies** - disable unnecessary detectors
4. **Configure appropriate caching** based on your use case
5. **Monitor and tune** connection pool settings
6. **Use edge caching** in distributed environments
7. **Implement regional routing** to reduce latency

Example performance configuration:

```python
from deepsentinel import SentinelConfig, PerformanceConfig

# Create optimized performance configuration
perf_config = PerformanceConfig(
    enable_caching=True,
    cache_ttl=300,            # 5 minutes
    cache_max_size=10000,     # Adjust based on memory availability
    connection_pool_size=100, # High-throughput settings
    connection_pool_per_host=20,
    enable_metrics=True,
    pattern_cache_size=500,   # Cache compiled regex patterns
    enable_async=True         # Use async operations where possible
)

config = SentinelConfig(performance_config=perf_config)
client = SentinelClient(config=config)
```

## Error Handling

### Q: How do I handle different types of errors?

**A:** Use specific exception types:

```python
from deepsentinel import (
    ComplianceViolationError,
    ProviderError, 
    AuthenticationError,
    RateLimitError
)

try:
    response = client.chat.completions.create(...)
except ComplianceViolationError as e:
    # Handle compliance issues
    print(f"Compliance violation: {e.violations}")
except AuthenticationError as e:
    # Handle authentication issues
    print(f"Auth error: {e.message}")
except RateLimitError as e:
    # Handle rate limiting
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except ProviderError as e:
    # Handle provider issues
    print(f"Provider {e.provider} error: {e.message}")
```

### Q: What should I do when I get a rate limit error?

**A:** Implement exponential backoff:

```python
import time
import random
from deepsentinel import RateLimitError

def make_request_with_retry(client, **kwargs):
    max_retries = 5
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(min(delay, 60))  # Cap at 60 seconds
```

### Q: How do I create a robust error handling strategy?

**A:** Implement a comprehensive error handling system:

```python
import logging
from deepsentinel import (
    SentinelClient, 
    ComplianceViolationError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError
)

logger = logging.getLogger(__name__)

class AIClient:
    def __init__(self):
        self.client = SentinelClient(...)
        
    def send_request(self, messages, **kwargs):
        """Send a request with robust error handling."""
        try:
            return self.client.chat.completions.create(
                messages=messages,
                **kwargs
            )
            
        except ComplianceViolationError as e:
            logger.warning(f"Compliance violation: {e.message}")
            # Return user-friendly message
            return self._format_error_response(
                "Your request contains sensitive information that cannot be processed."
            )
            
        except RateLimitError as e:
            logger.warning(f"Rate limited: {e.message}")
            # Implement retry mechanism
            return self._retry_with_backoff(messages, **kwargs)
            
        except ProviderError as e:
            logger.error(f"Provider error ({e.provider}): {e.message}")
            # Try with fallback provider
            if kwargs.get("provider") != "anthropic" and not kwargs.get("_is_fallback"):
                kwargs["provider"] = "anthropic"
                kwargs["_is_fallback"] = True
                return self.send_request(messages, **kwargs)
            return self._format_error_response("Service temporarily unavailable")
            
        except AuthenticationError:
            logger.critical("Authentication failed - check API keys")
            return self._format_error_response("Authentication error")
            
        except InvalidRequestError as e:
            logger.error(f"Invalid request: {e.message}")
            return self._format_error_response("Invalid request format")
            
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            return self._format_error_response("An unexpected error occurred")
            
    def _retry_with_backoff(self, messages, **kwargs):
        """Retry with exponential backoff."""
        # Implementation here...
        
    def _format_error_response(self, message):
        """Format consistent error response."""
        return {
            "error": True,
            "message": message
        }
```

## Streaming

### Q: Does DeepSentinel support streaming?

**A:** Yes! Streaming works the same as with provider SDKs:

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Q: How does compliance work with streaming?

**A:** DeepSentinel checks the request before streaming starts. Response content is monitored during streaming for any violations.

### Q: How do I handle errors in streaming responses?

**A:** Implement proper error handling for streaming:

```python
try:
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[...],
        stream=True
    )
    
    for chunk in stream:
        try:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            
except ComplianceViolationError as e:
    print(f"Compliance violation in request: {e.message}")
except Exception as e:
    print(f"Error initiating stream: {str(e)}")
```

## MCP Tools Integration

### Q: What are MCP tools?

**A:** Model Context Protocol (MCP) tools allow LLMs to interact with external systems through DeepSentinel's middleware. They enable capabilities like:

- File access and manipulation
- Web searches and content retrieval
- Database queries
- API interactions
- Complex data processing

### Q: How do I use MCP tools with DeepSentinel?

**A:** Configure MCP tools during client initialization:

```python
from deepsentinel import SentinelClient, MCPConfig, MCPTool

# Define custom tools
weather_tool = MCPTool(
    name="get_weather",
    description="Get weather information for a location",
    parameters={
        "location": {"type": "string", "description": "City or location name"},
        "days": {"type": "integer", "description": "Forecast days"}
    },
    handler=get_weather_function  # Your function that handles the tool
)

# Configure MCP
mcp_config = MCPConfig(
    tools=[weather_tool],
    enable_builtin_tools=True,  # Enable default tools
)

# Initialize client with MCP support
client = SentinelClient(
    sentinel_api_key="your-key",
    openai_api_key="your-openai-key",
    mcp_config=mcp_config
)
```

### Q: What built-in MCP tools are available?

**A:** DeepSentinel includes several built-in tools:

- **web_search**: Search the internet for information
- **file_reader**: Read file contents
- **database_query**: Run SQL queries against connected databases
- **api_request**: Make HTTP requests to external APIs
- **calculator**: Perform complex mathematical calculations

### Q: Can I integrate MCP tools with external services?

**A:** Yes, you can build integrations with various services:

```python
import requests
from deepsentinel import MCPTool

# Create a Jira ticket creation tool
def create_jira_ticket(summary, description, ticket_type="Bug"):
    # Implementation for API call to Jira
    response = requests.post(
        "https://your-jira-instance/rest/api/2/issue",
        json={
            "fields": {
                "project": {"key": "PROJECT"},
                "summary": summary,
                "description": description,
                "issuetype": {"name": ticket_type}
            }
        },
        auth=("username", "api_token")
    )
    return {"ticket_id": response.json()["id"]}

# Register as MCP tool
jira_tool = MCPTool(
    name="create_jira_ticket",
    description="Create a Jira ticket",
    parameters={
        "summary": {"type": "string", "description": "Ticket summary"},
        "description": {"type": "string", "description": "Detailed description"},
        "ticket_type": {"type": "string", "description": "Type of ticket", "default": "Bug"}
    },
    handler=create_jira_ticket
)

# Add to client configuration
client = SentinelClient(
    # ... other config
    mcp_config=MCPConfig(tools=[jira_tool])
)
```

## Development and Testing

### Q: How do I test my integration?

**A:** Create test cases for compliance detection:

```python
def test_compliance():
    # Test normal content (should work)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    assert response is not None
    
    # Test sensitive content (should be blocked)
    try:
        client.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "user", "content": "My SSN is 123-45-6789"}]
        )
        assert False, "Should have been blocked"
    except ComplianceViolationError:
        pass  # Expected
```

### Q: Can I use DeepSentinel in development without API keys?

**A:** Use mock mode for testing:

```python
config = SentinelConfig(
    mock_mode=True,  # Don't make real API calls
    mock_responses={"gpt-4o": "Mock response"}
)
```

### Q: How do I run unit tests with DeepSentinel?

**A:** Use the test fixtures and mocks provided:

```python
from deepsentinel.testing import MockSentinelClient

def test_your_function():
    # Create a mock client
    mock_client = MockSentinelClient()
    
    # Configure mock responses
    mock_client.add_response(
        model="gpt-4o",
        content="This is a mock response",
        tokens={"prompt": 10, "completion": 5}
    )
    
    # Configure violations
    mock_client.add_violation(
        pattern="credit card",
        violation_type="PCI",
        risk_score=0.95
    )
    
    # Test with mock
    response = mock_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    assert response.choices[0].message.content == "This is a mock response"
    
    # Test violation detection
    try:
        mock_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "My credit card number"}]
        )
        assert False, "Should have detected violation"
    except ComplianceViolationError as e:
        assert e.violations[0].type == "PCI"
```

## Pricing and Limits

### Q: How much does DeepSentinel cost?

**A:** DeepSentinel offers:

- **Free tier**: 1,000 requests/month
- **Pro tier**: Starting at $29/month
- **Enterprise**: Custom pricing

Visit [deepsentinel.ai/pricing](https://deepsentinel.ai/pricing) for current pricing.

### Q: Are there rate limits?

**A:** Rate limits depend on your plan:

- **Free**: 100 requests/hour
- **Pro**: 10,000 requests/hour  
- **Enterprise**: Custom limits

### Q: Do I still pay provider costs?

**A:** Yes, you pay both:

- **DeepSentinel**: For compliance and middleware services
- **LLM Provider**: For the actual AI model usage (OpenAI, Anthropic, etc.)

## Troubleshooting

### Q: My requests are being blocked unexpectedly

**A:** Check your compliance configuration:

1. **Review risk threshold**: Lower values are more strict
2. **Check detection settings**: Ensure appropriate detectors are enabled
3. **Test with known content**: Verify detection is working correctly
4. **Review logs**: Check audit logs for violation details

### Q: I'm seeing high latency

**A:** Optimize performance:

1. **Enable caching** to reuse compliance decisions
2. **Use local detection** for obvious patterns
3. **Tune risk thresholds** to reduce cloud analysis
4. **Monitor metrics** to identify bottlenecks

### Q: Authentication keeps failing

**A:** Verify your API keys:

1. **Check key format**: DeepSentinel keys start with `ds-`
2. **Verify permissions**: Ensure keys have proper permissions
3. **Check expiration**: Some keys may have expiration dates
4. **Test independently**: Test each provider's key separately

### Q: How do I resolve connection errors?

**A:** Common connection issues:

1. **Timeout errors**:
   - Increase timeout settings: `config = SentinelConfig(request_timeout=30)`
   - Check network connectivity and firewall settings

2. **SSL errors**:
   - Update CA certificates: `pip install --upgrade certifi`
   - Check for TLS interception in corporate environments

3. **DNS issues**:
   - Verify DNS resolution for API endpoints
   - Try alternate DNS servers if needed

### Q: DeepSentinel is working in development but not in production

**A:** Common production issues:

1. **Environment variables**: Verify they're correctly set in production
2. **Network restrictions**: Check if production environments have restricted outbound access
3. **Proxy settings**: Configure proxy if your production environment uses one:
   ```python
   config = SentinelConfig(
       proxies={
           "http": "http://proxy:8080",
           "https": "https://proxy:8080"
       }
   )
   ```
4. **Resource limits**: Ensure sufficient memory and CPU allocation

## Getting More Help

### Q: Where can I get additional support?

**A:** Multiple support channels available:

- **📖 Documentation**: [deepsentinel-ai.github.io/deepsentinel-python](https://deepsentinel-ai.github.io/deepsentinel-python)
- **💬 Discord**: [discord.gg/deepsentinel](https://discord.gg/deepsentinel)
- **🐛 GitHub Issues**: [GitHub Issues](https://github.com/deepsentinel/deepsentinel-sdk/issues)
- **📧 Email Support**: [support@deepsentinel.ai](mailto:support@deepsentinel.ai)
- **📞 Enterprise Support**: Available for enterprise customers

### Q: How do I report a bug?

**A:** Please include this information:

1. **DeepSentinel version**: `pip show deepsentinel`
2. **Python version**: `python --version`
3. **Error message**: Full stack trace
4. **Minimal reproduction**: Simplified code that reproduces the issue
5. **Expected vs actual behavior**

### Q: Can I contribute to DeepSentinel?

**A:** Yes! We welcome contributions:

- **Report bugs** and request features
- **Improve documentation** 
- **Submit pull requests** for bug fixes
- **Share usage examples** and tutorials

Check our [Contributing Guide](https://github.com/deepsentinel/deepsentinel-sdk/blob/main/CONTRIBUTING.md) for details.

---

**Still have questions?** Join our [Discord community](https://discord.gg/deepsentinel) or contact [support@deepsentinel.ai](mailto:support@deepsentinel.ai)