# Multiple Providers Tutorial

This tutorial shows you how to configure DeepSentinel to work with multiple LLM providers (OpenAI and Anthropic) and implement advanced routing strategies to optimize your application.

## Overview

By the end of this tutorial, you'll have:
- Set up DeepSentinel with multiple LLM providers
- Implemented provider selection strategies
- Created a simple routing system based on query content
- Added automatic failover between providers

**Time required:** 20 minutes

## Prerequisites

- Python 3.8+ installed
- DeepSentinel SDK installed (`pip install deepsentinel-sdk`)
- API keys for OpenAI and Anthropic
- Basic understanding of the [DeepSentinel basics](basic-integration.md)

## Step 1: Install and Configure Multiple Providers

First, make sure you have the DeepSentinel SDK installed:

```bash
pip install deepsentinel-sdk
```

Next, create a client with multiple provider configurations:

```python
from deepsentinel import SentinelClient

# Initialize with multiple provider API keys
client = SentinelClient(
    sentinel_api_key="your-deepsentinel-key",
    openai_api_key="your-openai-key",
    anthropic_api_key="your-anthropic-key"
)
```

## Step 2: Basic Provider Selection

The simplest way to use multiple providers is to explicitly specify which provider to use for each request:

```python
# Use OpenAI as the provider
openai_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing."}],
    provider="openai"  # Explicitly specify OpenAI
)

print(f"OpenAI Response: {openai_response.choices[0].message.content[:100]}...")

# Use Anthropic as the provider
anthropic_response = client.chat.completions.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "Explain quantum computing."}],
    provider="anthropic"  # Explicitly specify Anthropic
)

print(f"Anthropic Response: {anthropic_response.choices[0].message.content[:100]}...")
```

## Step 3: Provider Selection Based on Model

DeepSentinel can automatically select the appropriate provider based on the model you specify:

```python
# Automatically routes to OpenAI
gpt_response = client.chat.completions.create(
    model="gpt-4o",  # OpenAI model, automatically selects OpenAI provider
    messages=[{"role": "user", "content": "Write a haiku about AI."}]
)

# Automatically routes to Anthropic
claude_response = client.chat.completions.create(
    model="claude-3-opus-20240229",  # Anthropic model, automatically selects Anthropic
    messages=[{"role": "user", "content": "Write a haiku about AI."}]
)
```

## Step 4: Advanced Configuration with Provider-Specific Options

For more control, use the `SentinelConfig` to configure providers with specific settings:

```python
from deepsentinel import SentinelClient, SentinelConfig, ProviderConfig

# Create provider-specific configurations
config = SentinelConfig(
    sentinel_api_key="your-deepsentinel-key",
    default_provider="openai",  # Set the default provider
    provider_configs=[
        ProviderConfig(
            name="openai",
            api_key="your-openai-key",
            base_url="https://api.openai.com/v1",  # Default OpenAI API URL
            timeout=30.0,
            max_retries=3,
            model_aliases={
                "gpt-4-latest": "gpt-4o",  # Map shorthand names to actual models
                "gpt-3.5": "gpt-3.5-turbo"
            }
        ),
        ProviderConfig(
            name="anthropic",
            api_key="your-anthropic-key",
            base_url="https://api.anthropic.com",  # Default Anthropic API URL
            timeout=45.0,
            max_retries=2
        )
    ]
)

# Create client with the configuration
client = SentinelClient(config=config)

# Use with default provider (OpenAI)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Step 5: Implement Automatic Failover

Set up automatic failover between providers to increase reliability:

```python
from deepsentinel import SentinelClient, SentinelConfig

# Create configuration with failover settings
config = SentinelConfig(
    sentinel_api_key="your-deepsentinel-key",
    providers={
        "openai": {
            "api_key": "your-openai-key",
            "default_model": "gpt-4o"
        },
        "anthropic": {
            "api_key": "your-anthropic-key",
            "default_model": "claude-3-opus-20240229"
        }
    },
    fallback_strategy={
        "primary": "openai",        # Try OpenAI first
        "fallbacks": ["anthropic"],  # Fall back to Anthropic if OpenAI fails
        "auto_failover": True,       # Enable automatic failover
        "max_retries": 3,            # Number of retries before failing
        "retry_delay": 1,            # Seconds between retries
        "fallback_conditions": ["rate_limit", "timeout", "server_error"]
    }
)

# Initialize client with failover config
client = SentinelClient(config=config)

# This will try OpenAI first, then fall back to Anthropic if needed
try:
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": "What is the difference between OpenAI and Anthropic models?"
        }],
        # No need to specify model or provider - handled by fallback strategy
    )
    
    print(f"Response from: {response.model}")
    print(f"Content: {response.choices[0].message.content[:100]}...")
    
except Exception as e:
    print(f"All providers failed: {str(e)}")
```

## Step 6: Implement Content-Based Provider Routing

Implement a smart routing system that selects the most appropriate provider based on the content:

```python
def select_provider_by_use_case(content):
    """
    Select the most appropriate provider based on the content.
    
    Args:
        content: The user's query content
        
    Returns:
        Dict with provider and model to use
    """
    # Simple routing logic based on content
    content_lower = content.lower()
    
    if "code" in content_lower or "program" in content_lower:
        # OpenAI may be better for coding tasks
        return {"provider": "openai", "model": "gpt-4o"}
    
    elif "creative" in content_lower or "story" in content_lower:
        # Anthropic might be better for creative writing
        return {"provider": "anthropic", "model": "claude-3-opus-20240229"}
    
    elif len(content) > 500:
        # For longer prompts, Anthropic models might handle context better
        return {"provider": "anthropic", "model": "claude-3-opus-20240229"}
    
    else:
        # Default to OpenAI for general queries
        return {"provider": "openai", "model": "gpt-4o"}

# Example usage
user_query = "Write a function in Python to calculate the Fibonacci sequence."
selection = select_provider_by_use_case(user_query)
provider = selection["provider"]
model = selection["model"]

print(f"Selected provider: {provider}, model: {model}")

# Make the request with the selected provider
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": user_query}],
    provider=provider
)
```

## Step 7: Implement Load Balancing

For high-volume applications, you can implement load balancing across providers:

```python
from deepsentinel import SentinelConfig, LoadBalancingStrategy

config = SentinelConfig(
    sentinel_api_key="your-deepsentinel-key",
    providers={
        "openai": {"api_key": "your-openai-key"},
        "anthropic": {"api_key": "your-anthropic-key"}
    },
    load_balancing=LoadBalancingStrategy(
        strategy="round_robin",  # Options: "round_robin", "weighted", "least_latency"
        providers=["openai", "anthropic"],
        weights={"openai": 0.7, "anthropic": 0.3}  # For weighted strategy
    )
)

client = SentinelClient(config=config)

# Requests will be distributed according to the load balancing strategy
for i in range(5):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": f"Query {i}: What is AI?"}]
    )
    print(f"Query {i} handled by: {response.provider}")
```

## Step 8: Complete Implementation Example

Here's a complete implementation demonstrating multiple providers with advanced features:

```python
import os
from deepsentinel import SentinelClient, SentinelConfig, ProviderConfig

# Load API keys from environment variables
sentinel_api_key = os.getenv("DEEPSENTINEL_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Create configuration with multiple providers and advanced settings
config = SentinelConfig(
    sentinel_api_key=sentinel_api_key,
    default_provider="openai",
    provider_configs=[
        ProviderConfig(
            name="openai",
            api_key=openai_api_key,
            timeout=30.0,
            max_retries=3,
            rate_limit={
                "requests_per_minute": 3500,
                "tokens_per_minute": 90000
            }
        ),
        ProviderConfig(
            name="anthropic",
            api_key=anthropic_api_key, 
            timeout=45.0,
            max_retries=2,
            rate_limit={
                "requests_per_minute": 1000,
                "tokens_per_minute": 40000
            }
        )
    ],
    fallback_strategy={
        "primary": "openai",
        "fallbacks": ["anthropic"],
        "auto_failover": True,
        "max_retries": 3
    }
)

# Create client with advanced config
client = SentinelClient(config=config)

class MultiProviderRouter:
    """Helper class to route requests to the appropriate provider."""
    
    def __init__(self, client):
        self.client = client
    
    def query(self, user_input, **kwargs):
        """Send a query to the most appropriate provider."""
        selection = self._select_provider(user_input)
        provider = selection["provider"]
        model = selection["model"]
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_input}],
                provider=provider,
                **kwargs
            )
            
            print(f"Query handled by {provider} using {model}")
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error with primary provider {provider}: {str(e)}")
            
            # Try alternate provider
            alt_provider = "anthropic" if provider == "openai" else "openai"
            alt_model = "claude-3-opus-20240229" if alt_provider == "anthropic" else "gpt-4o"
            
            print(f"Trying alternate provider: {alt_provider}")
            response = self.client.chat.completions.create(
                model=alt_model,
                messages=[{"role": "user", "content": user_input}],
                provider=alt_provider,
                **kwargs
            )
            
            return response.choices[0].message.content
    
    def _select_provider(self, content):
        """Select the most appropriate provider based on content."""
        content_lower = content.lower()
        
        if "code" in content_lower or "program" in content_lower:
            return {"provider": "openai", "model": "gpt-4o"}
        elif "creative" in content_lower or "story" in content_lower:
            return {"provider": "anthropic", "model": "claude-3-opus-20240229"}
        elif len(content) > 500:
            return {"provider": "anthropic", "model": "claude-3-opus-20240229"}
        else:
            return {"provider": "openai", "model": "gpt-4o"}

# Usage
router = MultiProviderRouter(client)

# Example queries
queries = [
    "Explain how a neural network works.",
    "Write a creative short story about a robot learning to paint.",
    "Write a Python function to sort a list using merge sort.",
]

for query in queries:
    print(f"\nQuery: {query}")
    response = router.query(query, temperature=0.7)
    print(f"Response: {response[:150]}...")
```

## What's Next?

Congratulations! You've set up DeepSentinel with multiple providers and implemented advanced routing strategies. Here are some next steps to consider:

### Additional Optimizations
1. **Cost Optimization** - Route requests to cheaper providers for simpler tasks
2. **Performance Monitoring** - Track performance metrics for each provider
3. **Custom Adapters** - Create adapters for additional providers

### Advanced Topics
- **[Streaming Responses](streaming.md)** - Implement streaming with multiple providers
- **[Performance Optimization](../guides/performance.md)** - Optimize for high throughput
- **[Error Handling](error-handling.md)** - Robust error handling with multiple providers

## Common Issues

### Authentication Errors
```
AuthenticationError: Invalid API key for provider anthropic
```
**Solution**: Verify that your Anthropic API key is correct and has been properly configured.

### Provider Not Available
```
ProviderError: Provider 'anthropic' not configured
```
**Solution**: Ensure you've added all required providers to your configuration.

### Model Not Found
```
InvalidRequestError: Model 'claude-3-haiku-20240307' not found
```
**Solution**: Check that you're using the correct model names for each provider.

## Getting Help

- **[Provider Documentation](../concepts/providers.md)** - Learn more about provider integration
- **[API Reference](../reference/)** - Complete API documentation
- **[GitHub Issues](https://github.com/deepsentinel/deepsentinel-sdk/issues)** - Report bugs or request features

---

**Next Tutorial**: [Streaming Responses →](streaming.md)