# Basic Integration Tutorial

This tutorial shows you how to integrate DeepSentinel into an existing application that uses OpenAI's API. You'll learn how to add compliance checking with minimal code changes.

## Overview

By the end of this tutorial, you'll have:
- Replaced direct OpenAI API calls with DeepSentinel
- Added basic compliance checking for PII and sensitive data
- Implemented proper error handling for compliance violations
- Set up audit logging for all API interactions

**Time required:** 15 minutes

## Prerequisites

- Python 3.8+ installed
- An existing application using OpenAI's API
- DeepSentinel API key ([sign up here](https://deepsentinel.ai/signup))
- OpenAI API key

## Step 1: Install DeepSentinel

First, install the DeepSentinel SDK:

```bash
pip install deepsentinel-sdk```

## Step 2: Replace OpenAI Imports

**Before (using OpenAI directly):**
```python
import openai
from openai import OpenAI

client = OpenAI(api_key="your-openai-key")
```

**After (using DeepSentinel):**
```python
from deepsentinel import SentinelClient

client = SentinelClient(
    sentinel_api_key="your-deepsentinel-key",
    openai_api_key="your-openai-key"
)
```

## Step 3: Update API Calls

The great news is that DeepSentinel provides the same interface as OpenAI's SDK, so your existing code will work with minimal changes.

### Chat Completions

**Before:**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the best practices for data privacy?"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

**After:**
```python
# Exactly the same code!
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the best practices for data privacy?"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

### Streaming Responses

**Before:**
```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a short story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

**After:**
```python
# Same interface, now with compliance checking!
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a short story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## Step 4: Add Compliance Error Handling

Now let's add proper error handling for compliance violations:

```python
from deepsentinel import SentinelClient, ComplianceViolationError, ProviderError

client = SentinelClient(
    sentinel_api_key="your-deepsentinel-key",
    openai_api_key="your-openai-key"
)

def safe_chat_completion(messages, model="gpt-4o", **kwargs):
    """Make a chat completion with compliance error handling."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response
        
    except ComplianceViolationError as e:
        print(f"❌ Compliance violation detected: {e.message}")
        print(f"Violation types: {[v.type for v in e.violations]}")
        
        # Handle the violation (e.g., ask user to rephrase)
        return None
        
    except ProviderError as e:
        print(f"❌ Provider error: {e.message}")
        
        # Handle provider issues (e.g., retry with different provider)
        return None
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None

# Usage
response = safe_chat_completion([
    {"role": "user", "content": "Hello, world!"}
])

if response:
    print(response.choices[0].message.content)
else:
    print("Request failed due to compliance or technical issues")
```

## Step 5: Test Compliance Detection

Let's test that compliance detection is working by sending a request with sensitive data:

```python
# This should trigger a compliance violation
test_messages = [
    {
        "role": "user", 
        "content": "My credit card number is 4111-1111-1111-1111 and my SSN is 123-45-6789"
    }
]

print("Testing compliance detection...")
response = safe_chat_completion(test_messages)

if not response:
    print("✅ Compliance detection is working! Sensitive data was blocked.")
else:
    print("⚠️  Compliance detection may not be configured correctly.")
```

## Step 6: Configure Basic Compliance Policy

Create a basic compliance policy for your application:

```python
from deepsentinel import SentinelClient, CompliancePolicy

# Create a basic compliance policy
policy = CompliancePolicy(
    name="basic-app-policy",
    description="Basic compliance policy for our application",
    
    # Enable detection for common sensitive data types
    enable_pii_detection=True,
    enable_phi_detection=False,  # Disable if not handling health data
    enable_pci_detection=True,
    
    # Set risk tolerance
    max_risk_score=0.8,  # Block requests with risk score > 0.8
    block_on_violation=True,
    
    # Enable logging
    log_violations=True
)

# Initialize client with the policy
client = SentinelClient(
    sentinel_api_key="your-deepsentinel-key",
    openai_api_key="your-openai-key",
    compliance_policies=[policy]
)
```

## Step 7: Add Logging and Monitoring

Set up basic logging to track compliance events:

```python
import logging
from deepsentinel import SentinelClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def chat_with_logging(messages, **kwargs):
    """Chat completion with comprehensive logging."""
    try:
        logger.info(f"Making chat completion request with {len(messages)} messages")
        
        response = client.chat.completions.create(
            messages=messages,
            **kwargs
        )
        
        # Log successful completion
        logger.info(f"Chat completion successful. Tokens used: {response.usage.total_tokens}")
        return response
        
    except ComplianceViolationError as e:
        # Log compliance violations
        logger.warning(f"Compliance violation: {e.message}")
        logger.warning(f"Violation details: {[v.dict() for v in e.violations]}")
        raise
        
    except ProviderError as e:
        # Log provider errors
        logger.error(f"Provider error: {e.provider} - {e.message}")
        raise
        
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error in chat completion: {str(e)}")
        raise

# Usage with logging
try:
    response = chat_with_logging([
        {"role": "user", "content": "What's the weather like today?"}
    ])
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Request failed: {str(e)}")
```

## Step 8: Complete Integration Example

Here's a complete example showing a typical integration:

```python
import os
import logging
from typing import List, Dict, Optional
from deepsentinel import SentinelClient, CompliancePolicy, ComplianceViolationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    """Simple chatbot with DeepSentinel integration."""
    
    def __init__(self):
        # Initialize DeepSentinel client
        self.client = SentinelClient(
            sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            compliance_policies=[
                CompliancePolicy(
                    name="chatbot-policy",
                    enable_pii_detection=True,
                    enable_pci_detection=True,
                    max_risk_score=0.8,
                    block_on_violation=True
                )
            ],
            debug_mode=False
        )
        
        self.conversation_history = []
    
    def chat(self, user_message: str) -> Optional[str]:
        """Send a message and get a response."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user", 
            "content": user_message
        })
        
        try:
            # Make request with full conversation history
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    *self.conversation_history
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            # Extract response content
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            logger.info(f"Chat successful. Tokens used: {response.usage.total_tokens}")
            return assistant_message
            
        except ComplianceViolationError as e:
            error_msg = f"I can't process that message because it contains sensitive information: {', '.join([v.type for v in e.violations])}"
            logger.warning(f"Compliance violation: {e.message}")
            return error_msg
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return "Sorry, I'm having technical difficulties. Please try again."
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []

# Usage example
if __name__ == "__main__":
    bot = ChatBot()
    
    print("ChatBot with DeepSentinel Integration")
    print("Type 'quit' to exit, 'reset' to clear history")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            bot.reset_conversation()
            print("Conversation history cleared.")
            continue
        
        if user_input:
            response = bot.chat(user_input)
            print(f"Bot: {response}")
```

## Step 9: Test Your Integration

Create a test script to verify everything works:

```python
def test_integration():
    """Test the DeepSentinel integration."""
    print("Testing DeepSentinel integration...")
    
    # Test 1: Normal request should work
    print("\n1. Testing normal request...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, how are you?"}]
        )
        print("✅ Normal request successful")
    except Exception as e:
        print(f"❌ Normal request failed: {e}")
    
    # Test 2: Sensitive data should be blocked
    print("\n2. Testing compliance detection...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "My email is test@example.com"}]
        )
        print("⚠️  Sensitive data was not blocked (may be below risk threshold)")
    except ComplianceViolationError:
        print("✅ Compliance detection working - sensitive data blocked")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 3: Very sensitive data should definitely be blocked
    print("\n3. Testing high-risk content...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "My SSN is 123-45-6789 and credit card is 4111-1111-1111-1111"}]
        )
        print("❌ High-risk content was not blocked!")
    except ComplianceViolationError as e:
        print("✅ High-risk content properly blocked")
        print(f"   Detected: {[v.type for v in e.violations]}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_integration()
```

## What's Next?

Congratulations! You've successfully integrated DeepSentinel into your application. Here are some next steps:

### Immediate Next Steps
1. **Monitor compliance events** - Set up logging and monitoring for production
2. **Tune risk thresholds** - Adjust `max_risk_score` based on your requirements
3. **Add custom patterns** - Define organization-specific sensitive data patterns

### Advanced Features
- **[Multiple Providers](multiple-providers.md)** - Add Anthropic or other providers
- **[Streaming Responses](streaming.md)** - Implement real-time streaming
- **[Advanced Compliance](compliance-setup.md)** - Configure GDPR, HIPAA policies

### Production Considerations
- **Environment variables** - Use environment variables for API keys
- **Error monitoring** - Set up comprehensive error tracking
- **Performance monitoring** - Monitor latency and compliance decision times
- **Audit compliance** - Regular review of compliance logs and policies

## Common Issues

### Authentication Errors
```
AuthenticationError: Invalid API key
```
**Solution**: Verify your DeepSentinel and provider API keys are correct

### Import Errors
```
ImportError: No module named 'deepsentinel'
```
**Solution**: Install DeepSentinel: `pip install deepsentinel-sdk`

### No Compliance Detection
If sensitive data isn't being detected:
1. Check that detection is enabled in your policy
2. Verify the risk score threshold isn't too high
3. Test with obviously sensitive data (SSN, credit card numbers)

## Getting Help

- **[FAQ](../faq.md)** - Common questions and solutions
- **[API Reference](../reference/)** - Complete API documentation
- **[GitHub Issues](https://github.com/deepsentinel/deepsentinel-sdk/issues)** - Report bugs or request features
- **[Discord Community](https://discord.gg/deepsentinel)** - Get help from other developers

---

**Next Tutorial**: [Multiple Providers →](multiple-providers.md)