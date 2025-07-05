# Streaming Responses Tutorial

This tutorial shows you how to implement streaming responses with DeepSentinel. Streaming allows you to receive and display LLM responses in real-time as they're being generated, providing a more interactive user experience.

## Overview

By the end of this tutorial, you'll have:
- Implemented basic streaming responses
- Added real-time processing of streamed tokens
- Built an interactive chat interface with streaming
- Learned how compliance checking works with streams

**Time required:** 25 minutes

## Prerequisites

- Python 3.8+ installed
- DeepSentinel SDK installed (`pip install deepsentinel-sdk`)
- OpenAI or Anthropic API key
- Basic understanding of [DeepSentinel basics](basic-integration.md)

## Step 1: Set Up Basic Streaming

First, let's implement a simple streaming example:

```python
from deepsentinel import SentinelClient, ComplianceViolationError
import os
import sys
import time

# Initialize the client
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create a streaming request
try:
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user", 
            "content": "Write a short poem about AI safety"
        }],
        stream=True,  # Enable streaming
        temperature=0.7,
        max_tokens=200
    )
    
    # Process the stream
    print("Response: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            # Optional: add small delay to simulate typing effect
            time.sleep(0.01)
            
    print("\n\nStream completed")
    
except ComplianceViolationError as e:
    print(f"Compliance violation: {e.message}")
except Exception as e:
    print(f"Error: {str(e)}")
```

The key points in this example:
- Set `stream=True` in the `create` method
- Loop through each chunk in the returned stream
- Access `chunk.choices[0].delta.content` to get each piece of text as it arrives
- Use `end=""` and `flush=True` in print statements to display a continuous output

## Step 2: Token-by-Token Processing

You can process each token in the stream as it arrives for real-time effects:

```python
# Initialize the client
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Track tokens and words
word_count = 0
token_count = 0
current_word = ""

try:
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user", 
            "content": "Explain how neural networks process information"
        }],
        stream=True,
        max_tokens=200
    )
    
    print("Processing stream in real-time...")
    print("Response: ", end="")
    
    # Process each chunk in the stream
    for chunk in stream:
        if not chunk.choices[0].delta.content:
            continue
            
        content = chunk.choices[0].delta.content
        token_count += 1
        
        # Process content character by character
        for char in content:
            # Track word boundaries
            if char.isalnum() or char == "'":
                current_word += char
            elif current_word:  # Word boundary found
                word_count += 1
                current_word = ""
                
            # Print with color highlighting based on content
            if char in ".,!?;:":  # Highlight punctuation
                sys.stdout.write(f"\033[31m{char}\033[0m")  # Red
            elif char.isupper():  # Highlight uppercase
                sys.stdout.write(f"\033[34m{char}\033[0m")  # Blue
            else:
                sys.stdout.write(char)
                
            sys.stdout.flush()
            time.sleep(0.01)
    
    # Count the last word if content ends without punctuation
    if current_word:
        word_count += 1
        
    print("\n\nStream Statistics:")
    print(f"  Tokens received: {token_count}")
    print(f"  Words counted: {word_count}")
    print(f"  Approximate words per token: {word_count/token_count:.2f}")
    
except Exception as e:
    print(f"Error: {str(e)}")
```

This example demonstrates:
- Counting tokens and words in real-time
- Applying color highlighting to different character types
- Calculating streaming statistics

## Step 3: Progress Tracking for Long Responses

For longer responses, you can implement a progress bar or tracking mechanism:

```python
# Initialize the client
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

try:
    # Start a streaming request with a longer response
    print("Generating a story with progress tracking...")
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user", 
            "content": "Write a short story about a robot learning ethics"
        }],
        stream=True,
        max_tokens=500
    )
    
    # Variables for progress tracking
    start_time = time.time()
    token_count = 0
    estimated_tokens = 500  # Our max_tokens value
    response_text = ""
    
    # Track progress as tokens arrive
    print("Progress: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            token_count += 1
            response_text += content
            
            # Update progress bar
            progress = min(token_count / estimated_tokens, 1.0)
            bars = int(progress * 30)
            sys.stdout.write("\r")
            sys.stdout.write(
                f"Progress: [{'#' * bars}{' ' * (30 - bars)}] "
                f"{int(progress * 100)}% ({token_count}/{estimated_tokens})"
            )
            sys.stdout.flush()
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    tokens_per_second = token_count / elapsed_time
    
    # Print final statistics
    print("\n\nGeneration Statistics:")
    print(f"  Total tokens: {token_count}")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Speed: {tokens_per_second:.2f} tokens/second")
    
    # Print the full response
    print("\nFull story:")
    print(response_text)
    
except Exception as e:
    print(f"Error: {str(e)}")
```

This implementation shows:
- Creating a dynamic progress bar
- Tracking generation speed
- Recording the complete response as it builds

## Step 4: Building an Interactive Chat Interface

Now let's create an interactive chat application with streaming responses:

```python
from deepsentinel import SentinelClient, ComplianceViolationError
import os
import time

# Initialize the client
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize conversation history
conversation = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("Interactive Streaming Chat")
print("Type your messages and receive streaming responses.")
print("Type 'exit' or 'quit' to end the conversation.")

while True:
    # Get user input
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    
    # Add user message to conversation
    conversation.append({"role": "user", "content": user_input})
    
    try:
        # Start streaming response
        print("Assistant: ", end="")
        
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation,
            stream=True,
            temperature=0.7
        )
        
        # Collect assistant's response
        assistant_response = ""
        
        # Process the stream
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                assistant_response += content
                time.sleep(0.01)
        
        # Add assistant response to conversation history
        conversation.append({
            "role": "assistant",
            "content": assistant_response
        })
        
    except ComplianceViolationError as e:
        print(f"\nCompliance violation: {e.message}")
        print(f"Please avoid sending sensitive information.")
    except Exception as e:
        print(f"\nError: {str(e)}")
```

This example demonstrates:
- Maintaining conversation history
- Processing streaming responses in an interactive format
- Handling compliance violations

## Step 5: Compliance Checking with Streaming

DeepSentinel's compliance checking works differently with streaming responses. Let's explore how to handle compliance issues in streaming contexts:

```python
from deepsentinel import SentinelClient, ComplianceViolationError, CompliancePolicy
import os
import time

# Create a strict compliance policy
strict_policy = CompliancePolicy(
    name="strict-streaming-policy",
    description="Strict policy for streaming responses",
    enable_pii_detection=True,
    enable_pci_detection=True,
    max_risk_score=0.6,  # Lower threshold than default
    block_on_violation=True
)

# Initialize client with the strict policy
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    compliance_policies=[strict_policy]
)

def test_streaming_compliance():
    """Test how compliance checking works with streaming."""
    print("Testing streaming compliance checks...")
    
    # Example 1: Safe content
    print("\n1. Testing with safe content...")
    try:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": "Write a short poem about mountains"
            }],
            stream=True
        )
        
        print("Response: ", end="")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                time.sleep(0.01)
                
        print("\n✅ Safe content successfully streamed")
        
    except ComplianceViolationError as e:
        print(f"\n❌ Unexpected compliance violation: {e.message}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    # Example 2: Content with PII
    print("\n2. Testing with sensitive content...")
    try:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": "My social security number is 123-45-6789"
            }],
            stream=True
        )
        
        print("Response: ", end="")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                time.sleep(0.01)
                
        print("\n⚠️ Sensitive content was not blocked (unexpected)")
        
    except ComplianceViolationError as e:
        print(f"\n✅ Compliance violation correctly detected: {e.message}")
        print(f"Violation types: {[v.type for v in e.violations]}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

# Run the compliance test
test_streaming_compliance()
```

Note: For streaming responses, compliance checking happens before streaming begins. If a compliance violation is detected in the prompt, the stream won't start and will raise a `ComplianceViolationError`. For responses, DeepSentinel monitors the stream and terminates it if a violation is detected.

## Step 6: Streaming with Multiple Providers

You can use streaming with different LLM providers:

```python
from deepsentinel import SentinelClient
import os
import time

# Initialize client with multiple providers
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

def stream_from_provider(provider, model, prompt):
    """Stream a response from a specific provider."""
    print(f"\nStreaming from {provider} ({model}):")
    print("-" * 40)
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            stream=True,
            max_tokens=150
        )
        
        print("Response: ", end="")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                time.sleep(0.01)
                
        print("\n")
        
    except Exception as e:
        print(f"Error with {provider}: {str(e)}")

# Stream from both providers with the same prompt
prompt = "Explain the concept of streaming in AI in three sentences."
stream_from_provider("openai", "gpt-4o", prompt)
stream_from_provider("anthropic", "claude-3-opus-20240229", prompt)
```

This example demonstrates:
- Using the same streaming interface across different providers
- Comparing streaming behavior between providers

## Step 7: Error Handling for Streams

Let's implement robust error handling for streaming responses:

```python
from deepsentinel import (
    SentinelClient, 
    ComplianceViolationError,
    ProviderError,
    RateLimitError
)
import os
import time

# Initialize the client
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def stream_with_error_handling(messages, model="gpt-4o", **kwargs):
    """Stream a response with comprehensive error handling."""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        print("Response: ", end="")
        response_text = ""
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                response_text += content
                time.sleep(0.01)
                
        print("\n")
        return {"success": True, "content": response_text}
        
    except ComplianceViolationError as e:
        print(f"\nCompliance violation: {e.message}")
        return {
            "success": False, 
            "error_type": "compliance",
            "message": e.message,
            "violations": [v.type for v in e.violations]
        }
        
    except RateLimitError as e:
        print(f"\nRate limit exceeded. Retry after {e.retry_after} seconds.")
        return {
            "success": False,
            "error_type": "rate_limit",
            "retry_after": e.retry_after
        }
        
    except ProviderError as e:
        print(f"\nProvider error ({e.provider}): {e.message}")
        return {
            "success": False,
            "error_type": "provider",
            "provider": e.provider,
            "message": e.message
        }
        
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return {
            "success": False,
            "error_type": "unknown",
            "message": str(e)
        }

# Test error handling with different cases
test_cases = [
    # Normal case
    [{"role": "user", "content": "Hello, how are you?"}],
    
    # Potentially triggering compliance
    [{"role": "user", "content": "My credit card is 4111-1111-1111-1111"}],
    
    # Invalid model (will cause provider error)
    [{"role": "user", "content": "Hello"}], 
]

for i, messages in enumerate(test_cases):
    print(f"\nTest case {i+1}:")
    
    # For the last test, use an invalid model
    model = "non-existent-model" if i == 2 else "gpt-4o"
    
    result = stream_with_error_handling(messages, model=model)
    
    if result["success"]:
        print(f"Success! Generated {len(result['content'])} characters")
    else:
        print(f"Failed with {result['error_type']} error: {result.get('message', '')}")
```

This implementation provides:
- Structured error handling for all stream-related errors
- Detailed error reporting
- Options for error recovery

## Step 8: Complete Implementation Example

Let's put everything together in a comprehensive streaming chat application:

```python
import os
import sys
import time
from typing import List, Dict, Optional, Any
from deepsentinel import SentinelClient, ComplianceViolationError

class StreamingChatApp:
    """Interactive streaming chat application with DeepSentinel."""
    
    def __init__(self, api_keys: Dict[str, str], model: str = "gpt-4o"):
        """Initialize the chat application.
        
        Args:
            api_keys: Dictionary with sentinel_api_key and provider keys
            model: Default model to use
        """
        self.client = SentinelClient(**api_keys)
        self.model = model
        self.conversation = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.token_count = 0
        self.response_time = 0
    
    def chat(self, message: str) -> Dict[str, Any]:
        """Send a message and stream the response.
        
        Args:
            message: User message to send
            
        Returns:
            Dictionary with results
        """
        # Add user message to history
        self.conversation.append({"role": "user", "content": message})
        
        try:
            # Start timing
            start_time = time.time()
            
            # Create streaming request
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                stream=True,
                temperature=0.7
            )
            
            # Process stream
            print("Assistant: ", end="")
            response_content = ""
            tokens = 0
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    response_content += content
                    tokens += 1
                    time.sleep(0.01)
            
            # Record metrics
            self.response_time = time.time() - start_time
            self.token_count += tokens
            
            # Add response to conversation history
            self.conversation.append({
                "role": "assistant",
                "content": response_content
            })
            
            return {
                "success": True,
                "content": response_content,
                "tokens": tokens,
                "response_time": self.response_time
            }
            
        except ComplianceViolationError as e:
            print(f"\nCompliance violation: {e.message}")
            return {
                "success": False,
                "error": "compliance_violation",
                "message": e.message
            }
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            return {
                "success": False,
                "error": "general_error",
                "message": str(e)
            }
    
    def display_stats(self):
        """Display chat statistics."""
        messages = len(self.conversation) - 1  # Exclude system message
        user_messages = messages // 2
        
        print("\n----- Chat Statistics -----")
        print(f"Messages exchanged: {user_messages} user, {user_messages} assistant")
        print(f"Approximate tokens: {self.token_count}")
        print(f"Last response time: {self.response_time:.2f} seconds")
    
    def reset(self):
        """Reset the conversation."""
        self.conversation = [self.conversation[0]]  # Keep system message
        self.token_count = 0
        print("Conversation has been reset.")

def main():
    """Run the streaming chat application."""
    # Get API keys from environment
    api_keys = {
        "sentinel_api_key": os.getenv("DEEPSENTINEL_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }
    
    # Initialize chat app
    app = StreamingChatApp(api_keys)
    
    print("🚀 DeepSentinel Streaming Chat")
    print("=" * 40)
    print("Type your messages and receive streaming responses.")
    print("Commands: /stats - Show statistics, /reset - Reset conversation, /exit - Exit")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Handle commands
        if user_input.lower() in ["/exit", "/quit"]:
            print("Goodbye!")
            break
            
        elif user_input.lower() == "/stats":
            app.display_stats()
            continue
            
        elif user_input.lower() == "/reset":
            app.reset()
            continue
        
        # Process regular messages
        if user_input:
            result = app.chat(user_input)
            print()  # Add newline after response

if __name__ == "__main__":
    main()
```

This complete application provides:
- An interactive streaming chat interface
- Command handling for statistics and conversation reset
- Comprehensive error handling
- Performance tracking

## Performance Considerations for Streaming

Streaming responses have different performance characteristics compared to non-streaming:

1. **Lower perceived latency** - Users see results immediately, even if total completion time is the same
2. **Increased connection time** - Streaming keeps connections open longer
3. **Potential timeout issues** - Long streams may hit timeout limits
4. **Compliance checking overhead** - Real-time scanning adds complexity

To optimize streaming performance:

```python
# Configure optimized settings for streaming
from deepsentinel import SentinelClient, SentinelConfig, PerformanceConfig

# Create performance-optimized config
perf_config = PerformanceConfig(
    connection_pool_size=50,  # Increase connection pool for parallel streams
    connection_timeout=10.0,  # Longer timeout for streaming connections
    read_timeout=60.0,        # Extended read timeout for long streams
    keep_alive=True,          # Keep connections alive between requests
)

config = SentinelConfig(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    performance_config=perf_config
)

# Create client with optimized config
client = SentinelClient(config=config)
```

## What's Next?

Congratulations! You've implemented streaming responses with DeepSentinel. Here are some ways to build on what you've learned:

### Advanced Techniques
1. **Front-end Integration** - Connect these streams to a web UI using WebSockets
2. **Multi-stream Processing** - Process multiple parallel streams for batch operations
3. **Stream Filtering** - Implement real-time content filtering or transformation

### Next Topics to Explore
- **[Compliance Setup](compliance-setup.md)** - Detailed compliance configuration
- **[Multiple Providers](multiple-providers.md)** - Using multiple providers with streaming
- **[Error Handling](error-handling.md)** - More robust error recovery strategies

## Common Issues

### Stream Disconnects
```
ConnectionError: Connection closed while streaming response
```
**Solution**: Increase timeouts and implement reconnection logic

### Compliance Blocks
```
ComplianceViolationError: Sensitive content detected in response
```
**Solution**: Adjust compliance settings or pre-process user inputs

### Buffer Overruns
```
MemoryError: Buffer overflow while collecting stream
```
**Solution**: Process stream chunks individually instead of building a complete response

## Getting Help

- **[FAQ](../faq.md)** - Common questions and solutions
- **[Performance Guide](../guides/performance.md)** - Optimizing streaming performance
- **[API Reference](../reference/)** - Complete API documentation

---

**Next Tutorial**: [Compliance Setup →](compliance-setup.md)