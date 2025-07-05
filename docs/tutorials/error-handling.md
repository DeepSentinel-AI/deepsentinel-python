# Error Handling Tutorial

This tutorial shows you how to build robust applications with DeepSentinel by implementing comprehensive error handling patterns for various types of failures including compliance violations, provider errors, and network issues.

## Overview

By the end of this tutorial, you'll have:
- Implemented comprehensive error handling for all DeepSentinel error types
- Created recovery strategies for different failure scenarios
- Built resilient applications with proper fallback mechanisms
- Set up monitoring and alerting for error conditions

**Time required:** 25 minutes

## Prerequisites

- Python 3.8+ installed
- DeepSentinel SDK installed (`pip install deepsentinel-sdk`)
- OpenAI or Anthropic API key
- Basic understanding of [DeepSentinel basics](basic-integration.md)

## Step 1: Understanding DeepSentinel Error Types

DeepSentinel provides specific error types for different failure scenarios:

```python
from deepsentinel import (
    SentinelClient,
    ComplianceViolationError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    NetworkError,
    TimeoutError
)
```

Let's understand each error type:

- **`ComplianceViolationError`** - Sensitive data detected in request/response
- **`AuthenticationError`** - Invalid API keys or authentication failures
- **`ProviderError`** - LLM provider-specific errors (OpenAI, Anthropic, etc.)
- **`RateLimitError`** - Rate limits exceeded for provider or DeepSentinel
- **`InvalidRequestError`** - Malformed requests or invalid parameters
- **`NetworkError`** - Network connectivity issues
- **`TimeoutError`** - Request timeouts

## Step 2: Basic Error Handling Function

Let's create a comprehensive error handling function:

```python
from deepsentinel import (
    SentinelClient,
    ComplianceViolationError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    NetworkError,
    TimeoutError
)
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_completion(client, messages, model="gpt-4o", **kwargs):
    """
    Make a chat completion request with comprehensive error handling.
    
    Args:
        client: The SentinelClient instance
        messages: List of message objects
        model: Model to use for completion
        **kwargs: Additional parameters to pass to the API
    
    Returns:
        Dict with success status, data, or error information
    """
    try:
        # Attempt to create a chat completion
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        logger.info(f"Request successful. Model: {response.model}, Tokens: {response.usage.total_tokens}")
        
        return {
            "success": True,
            "data": response,
            "content": response.choices[0].message.content,
            "model": response.model,
            "tokens": response.usage.total_tokens
        }
    
    except ComplianceViolationError as e:
        # Handle compliance violations (PII, PCI, PHI, etc.)
        logger.warning(f"Compliance violation: {e.message}, Risk score: {e.risk_score}")
        
        return {
            "success": False,
            "error_type": "compliance_violation",
            "message": f"Request blocked due to sensitive data: {e.message}",
            "violations": [v.type for v in e.violations],
            "risk_score": e.risk_score,
            "recoverable": False,  # Usually not recoverable without user intervention
            "suggested_action": "Remove sensitive information and try again"
        }
    
    except AuthenticationError as e:
        # Handle authentication issues (invalid API keys)
        logger.error(f"Authentication error: {e.message}")
        
        return {
            "success": False,
            "error_type": "authentication",
            "message": f"Authentication failed: {e.message}",
            "recoverable": False,  # Requires fixing API keys
            "suggested_action": "Verify your API keys are correct and valid"
        }
    
    except RateLimitError as e:
        # Handle rate limiting (too many requests)
        logger.warning(f"Rate limit exceeded: {e.message}, Retry after: {e.retry_after}s")
        
        return {
            "success": False,
            "error_type": "rate_limit",
            "message": f"Rate limit exceeded: {e.message}",
            "retry_after": e.retry_after,
            "recoverable": True,  # Can retry after waiting
            "suggested_action": f"Wait {e.retry_after} seconds and retry"
        }
    
    except ProviderError as e:
        # Handle provider-specific errors (OpenAI, Anthropic)
        logger.error(f"Provider error ({e.provider}): {e.message}, Status: {e.status_code}")
        
        # Determine if error is temporary or permanent
        recoverable = e.status_code in [429, 500, 502, 503, 504]
        
        return {
            "success": False,
            "error_type": "provider",
            "message": f"Provider error ({e.provider}): {e.message}",
            "provider": e.provider,
            "status_code": e.status_code,
            "recoverable": recoverable,
            "suggested_action": "Try again later" if recoverable else "Check request parameters"
        }
    
    except InvalidRequestError as e:
        # Handle malformed requests
        logger.error(f"Invalid request: {e.message}")
        
        return {
            "success": False,
            "error_type": "invalid_request",
            "message": f"Invalid request: {e.message}",
            "recoverable": False,  # Requires fixing the request
            "suggested_action": "Check your request parameters and format"
        }
    
    except (NetworkError, TimeoutError) as e:
        # Handle network and timeout issues
        logger.warning(f"Network/timeout error: {e.message}")
        
        return {
            "success": False,
            "error_type": "network_timeout",
            "message": f"Network/timeout error: {e.message}",
            "recoverable": True,  # Usually temporary
            "suggested_action": "Check network connection and retry"
        }
    
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error: {str(e)}")
        
        return {
            "success": False,
            "error_type": "unexpected",
            "message": f"Unexpected error: {str(e)}",
            "recoverable": True,  # Unknown, so assume it might be temporary
            "suggested_action": "Try again or contact support"
        }
```

## Step 3: Implementing Retry Logic

For recoverable errors, implement intelligent retry logic:

```python
import time
import random
from typing import Optional, Callable

def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay."""
    delay = base_delay * (2 ** attempt)
    # Add jitter to prevent thundering herd
    jitter = random.uniform(0.1, 0.5) * delay
    return min(delay + jitter, max_delay)

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    backoff_strategy: str = "exponential",
    base_delay: float = 1.0,
    recoverable_errors: Optional[list] = None
):
    """
    Retry a function with backoff strategy.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        backoff_strategy: "exponential", "linear", or "fixed"
        base_delay: Base delay in seconds
        recoverable_errors: List of recoverable error types
    """
    if recoverable_errors is None:
        recoverable_errors = ["rate_limit", "network_timeout", "provider"]
    
    for attempt in range(max_retries + 1):
        result = func()
        
        # If successful, return immediately
        if result["success"]:
            if attempt > 0:
                logger.info(f"Request succeeded after {attempt} retries")
            return result
        
        # If not recoverable, don't retry
        if not result.get("recoverable", False):
            logger.info(f"Error not recoverable: {result['error_type']}")
            return result
        
        # If we've exhausted retries, return the last error
        if attempt >= max_retries:
            logger.error(f"Max retries ({max_retries}) exceeded")
            return result
        
        # Calculate delay for next attempt
        if backoff_strategy == "exponential":
            delay = exponential_backoff(attempt, base_delay)
        elif backoff_strategy == "linear":
            delay = base_delay * (attempt + 1)
        else:  # fixed
            delay = base_delay
        
        # Special handling for rate limits
        if result["error_type"] == "rate_limit" and "retry_after" in result:
            delay = max(delay, result["retry_after"])
        
        logger.info(f"Retry attempt {attempt + 1} in {delay:.2f} seconds...")
        time.sleep(delay)
    
    return result

# Usage example
def make_request_with_retry():
    """Make a request with automatic retry logic."""
    client = SentinelClient(
        sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    def _request():
        return safe_completion(
            client,
            [{"role": "user", "content": "What are best practices for error handling?"}],
            model="gpt-4o"
        )
    
    return retry_with_backoff(
        _request,
        max_retries=3,
        backoff_strategy="exponential",
        base_delay=1.0
    )

# Test the retry logic
result = make_request_with_retry()
if result["success"]:
    print(f"Success: {result['content'][:100]}...")
else:
    print(f"Failed after retries: {result['message']}")
```

## Step 4: Provider Failover Implementation

When using multiple providers, implement automatic failover:

```python
from deepsentinel import SentinelClient
import os

class MultiProviderClient:
    """Client with automatic provider failover."""
    
    def __init__(self, api_keys: dict, provider_priority: list = None):
        """
        Initialize client with multiple providers.
        
        Args:
            api_keys: Dict with API keys for different services
            provider_priority: List of providers in order of preference
        """
        self.client = SentinelClient(**api_keys)
        self.provider_priority = provider_priority or ["openai", "anthropic"]
        self.provider_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-opus-20240229"
        }
        
    def completion_with_failover(self, messages, **kwargs):
        """Make completion with automatic provider failover."""
        last_error = None
        
        for provider in self.provider_priority:
            model = self.provider_models.get(provider)
            if not model:
                continue
                
            logger.info(f"Trying provider: {provider}")
            
            result = safe_completion(
                self.client,
                messages,
                model=model,
                provider=provider,
                **kwargs
            )
            
            # If successful, return immediately
            if result["success"]:
                logger.info(f"Request successful with provider: {provider}")
                return result
            
            # Store the error for potential return
            last_error = result
            
            # If it's a non-recoverable error specific to the request
            # (not the provider), don't try other providers
            if result["error_type"] in ["compliance_violation", "invalid_request"]:
                logger.info(f"Non-recoverable error, not trying other providers: {result['error_type']}")
                break
            
            # If it's a provider-specific error, try the next provider
            if result["error_type"] in ["authentication", "provider", "rate_limit"]:
                logger.warning(f"Provider {provider} failed: {result['message']}")
                continue
        
        # All providers failed
        logger.error("All providers failed")
        return last_error or {"success": False, "error_type": "all_providers_failed", "message": "All providers failed"}

# Usage example
def test_provider_failover():
    """Test provider failover functionality."""
    api_keys = {
        "sentinel_api_key": os.getenv("DEEPSENTINEL_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY")
    }
    
    multi_client = MultiProviderClient(api_keys, provider_priority=["openai", "anthropic"])
    
    # Test with normal request
    result = multi_client.completion_with_failover([
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ])
    
    if result["success"]:
        print(f"✅ Success with provider failover")
        print(f"Response: {result['content'][:100]}...")
    else:
        print(f"❌ All providers failed: {result['message']}")

test_provider_failover()
```

## Step 5: Compliance Violation Recovery

Handle compliance violations gracefully by offering recovery options:

```python
from deepsentinel import ComplianceViolationError
import re

class ComplianceRecoveryHandler:
    """Handler for compliance violation recovery."""
    
    def __init__(self, client):
        self.client = client
        
    def sanitize_input(self, text: str, violations: list) -> str:
        """
        Attempt to sanitize input by removing detected sensitive data.
        
        Args:
            text: Original text with violations
            violations: List of compliance violations
            
        Returns:
            Sanitized text
        """
        sanitized = text
        
        for violation in violations:
            violation_type = violation.type
            
            if violation_type == "ssn":
                # Remove SSN patterns
                sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REMOVED]', sanitized)
                sanitized = re.sub(r'\b\d{9}\b', '[SSN REMOVED]', sanitized)
                
            elif violation_type == "credit_card":
                # Remove credit card patterns
                sanitized = re.sub(r'\b(?:\d{4}[- ]){3}\d{4}\b', '[CARD REMOVED]', sanitized)
                sanitized = re.sub(r'\b\d{16}\b', '[CARD REMOVED]', sanitized)
                
            elif violation_type == "email":
                # Remove email addresses
                sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REMOVED]', sanitized)
                
            elif violation_type == "phone":
                # Remove phone numbers
                sanitized = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE REMOVED]', sanitized)
                sanitized = re.sub(r'\(\d{3}\)\s*\d{3}-\d{4}', '[PHONE REMOVED]', sanitized)
        
        return sanitized
    
    def handle_compliance_violation(self, messages, violation_error):
        """
        Handle compliance violations with recovery options.
        
        Args:
            messages: Original messages that caused violation
            violation_error: ComplianceViolationError instance
            
        Returns:
            Dict with recovery options and results
        """
        violations = violation_error.violations
        risk_score = violation_error.risk_score
        
        print(f"🚨 Compliance violation detected!")
        print(f"Risk score: {risk_score:.2f}")
        print(f"Violations: {[v.type for v in violations]}")
        
        # Offer recovery options
        print("\nRecovery options:")
        print("1. Sanitize input automatically")
        print("2. Edit input manually")
        print("3. Cancel request")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == "1":
            # Automatic sanitization
            original_content = messages[-1]["content"]  # Assuming last message has the violation
            sanitized_content = self.sanitize_input(original_content, violations)
            
            print(f"\nOriginal: {original_content}")
            print(f"Sanitized: {sanitized_content}")
            
            # Create new messages with sanitized content
            new_messages = messages[:-1] + [{"role": "user", "content": sanitized_content}]
            
            # Retry with sanitized input
            result = safe_completion(self.client, new_messages)
            
            return {
                "recovery_method": "automatic_sanitization",
                "result": result,
                "sanitized_input": sanitized_content
            }
            
        elif choice == "2":
            # Manual editing
            original_content = messages[-1]["content"]
            print(f"\nOriginal content: {original_content}")
            new_content = input("Enter corrected content: ").strip()
            
            if new_content:
                new_messages = messages[:-1] + [{"role": "user", "content": new_content}]
                result = safe_completion(self.client, new_messages)
                
                return {
                    "recovery_method": "manual_edit",
                    "result": result,
                    "edited_input": new_content
                }
            else:
                return {
                    "recovery_method": "cancelled",
                    "result": {"success": False, "message": "No input provided"}
                }
        
        else:
            return {
                "recovery_method": "cancelled",
                "result": {"success": False, "message": "Request cancelled by user"}
            }

# Usage example
def interactive_compliance_recovery():
    """Interactive example of compliance violation recovery."""
    client = SentinelClient(
        sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    recovery_handler = ComplianceRecoveryHandler(client)
    
    # Test with content that will trigger compliance violation
    test_messages = [
        {"role": "user", "content": "My SSN is 123-45-6789 and email is john@example.com"}
    ]
    
    try:
        result = safe_completion(client, test_messages)
        
        if result["success"]:
            print("✅ Request successful")
            print(f"Response: {result['content'][:100]}...")
        
    except ComplianceViolationError as e:
        print("Compliance violation occurred, attempting recovery...")
        recovery_result = recovery_handler.handle_compliance_violation(test_messages, e)
        
        if recovery_result["result"]["success"]:
            print(f"✅ Recovery successful using {recovery_result['recovery_method']}")
            print(f"Response: {recovery_result['result']['content'][:100]}...")
        else:
            print(f"❌ Recovery failed: {recovery_result['result']['message']}")

# Run the interactive example
# interactive_compliance_recovery()
```

## Step 6: Monitoring and Alerting

Set up comprehensive monitoring for error conditions:

```python
import logging
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List

class ErrorMonitor:
    """Monitor and track errors for alerting and analysis."""
    
    def __init__(self, alert_thresholds: Dict = None):
        """
        Initialize error monitor.
        
        Args:
            alert_thresholds: Dict with error type -> threshold mappings
        """
        self.error_history = []
        self.error_counts = Counter()
        self.alert_thresholds = alert_thresholds or {
            "compliance_violation": 10,  # Alert after 10 violations in window
            "rate_limit": 5,
            "provider": 3,
            "authentication": 1,  # Alert immediately for auth errors
            "network_timeout": 15
        }
        self.alert_window = timedelta(minutes=10)  # 10-minute window
        
        # Set up logging
        self.logger = logging.getLogger("error_monitor")
        handler = logging.FileHandler("error_monitor.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def record_error(self, error_result: Dict):
        """Record an error for monitoring."""
        if error_result.get("success"):
            return  # Not an error
        
        error_record = {
            "timestamp": datetime.now(),
            "error_type": error_result.get("error_type"),
            "message": error_result.get("message"),
            "recoverable": error_result.get("recoverable"),
            "provider": error_result.get("provider"),
            "status_code": error_result.get("status_code")
        }
        
        self.error_history.append(error_record)
        self.error_counts[error_record["error_type"]] += 1
        
        # Log the error
        self.logger.error(f"Error recorded: {json.dumps(error_record, default=str)}")
        
        # Check if we need to send alerts
        self._check_alerts(error_record)
    
    def _check_alerts(self, error_record: Dict):
        """Check if error thresholds are exceeded and send alerts."""
        error_type = error_record["error_type"]
        threshold = self.alert_thresholds.get(error_type, float('inf'))
        
        # Count errors of this type in the alert window
        cutoff_time = datetime.now() - self.alert_window
        recent_errors = [
            e for e in self.error_history 
            if e["timestamp"] > cutoff_time and e["error_type"] == error_type
        ]
        
        if len(recent_errors) >= threshold:
            self._send_alert(error_type, len(recent_errors), recent_errors)
    
    def _send_alert(self, error_type: str, count: int, recent_errors: List):
        """Send alert for error threshold breach."""
        alert_message = f"🚨 ALERT: {error_type} errors exceeded threshold"
        alert_details = f"Count: {count} in last {self.alert_window.total_seconds()/60:.0f} minutes"
        
        # Log alert
        self.logger.critical(f"{alert_message} - {alert_details}")
        
        # In a real implementation, you would send this to:
        # - Slack/Teams webhook
        # - Email notification service
        # - PagerDuty or similar alerting system
        # - Monitoring dashboard
        
        print(f"\n{alert_message}")
        print(f"{alert_details}")
        print("Recent errors:")
        for error in recent_errors[-5:]:  # Show last 5 errors
            print(f"  - {error['timestamp']}: {error['message']}")
    
    def get_error_summary(self, hours: int = 24) -> Dict:
        """Get error summary for the specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e["timestamp"] > cutoff_time]
        
        error_summary = defaultdict(int)
        provider_summary = defaultdict(int)
        recoverable_summary = {"recoverable": 0, "non_recoverable": 0}
        
        for error in recent_errors:
            error_summary[error["error_type"]] += 1
            if error.get("provider"):
                provider_summary[error["provider"]] += 1
            
            if error.get("recoverable"):
                recoverable_summary["recoverable"] += 1
            else:
                recoverable_summary["non_recoverable"] += 1
        
        return {
            "time_window_hours": hours,
            "total_errors": len(recent_errors),
            "error_types": dict(error_summary),
            "provider_errors": dict(provider_summary),
            "recoverability": recoverable_summary,
            "error_rate": len(recent_errors) / max(1, hours)  # errors per hour
        }

# Integration with our safe_completion function
def monitored_completion(client, messages, monitor: ErrorMonitor, **kwargs):
    """Make completion with error monitoring."""
    result = safe_completion(client, messages, **kwargs)
    
    # Record error if unsuccessful
    if not result["success"]:
        monitor.record_error(result)
    
    return result

# Usage example
def test_error_monitoring():
    """Test error monitoring functionality."""
    client = SentinelClient(
        sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Set up monitoring with custom thresholds
    monitor = ErrorMonitor(alert_thresholds={
        "compliance_violation": 2,  # Alert after 2 violations for testing
        "rate_limit": 3
    })
    
    # Test various scenarios
    test_scenarios = [
        # Normal request
        [{"role": "user", "content": "Hello, how are you?"}],
        
        # Compliance violation
        [{"role": "user", "content": "My SSN is 123-45-6789"}],
        
        # Another compliance violation (should trigger alert)
        [{"role": "user", "content": "Credit card: 4111-1111-1111-1111"}],
        
        # Invalid model (provider error)
        [{"role": "user", "content": "Hello"}],  # We'll use invalid model for this
    ]
    
    for i, messages in enumerate(test_scenarios):
        print(f"\nTest scenario {i+1}:")
        
        # Use invalid model for the last test
        model = "non-existent-model" if i == 3 else "gpt-4o"
        
        result = monitored_completion(client, messages, monitor, model=model)
        
        if result["success"]:
            print(f"✅ Success: {result['content'][:50]}...")
        else:
            print(f"❌ Error: {result['message']}")
    
    # Print error summary
    print("\n" + "="*50)
    print("ERROR SUMMARY")
    print("="*50)
    summary = monitor.get_error_summary(1)  # Last 1 hour
    print(f"Total errors in last hour: {summary['total_errors']}")
    print(f"Error types: {summary['error_types']}")
    print(f"Error rate: {summary['error_rate']:.2f} errors/hour")

# Run the monitoring test
# test_error_monitoring()
```

## Step 7: Circuit Breaker Pattern

Implement a circuit breaker to prevent cascading failures:

```python
import time
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered

class CircuitBreaker:
    """Circuit breaker implementation for protecting against cascading failures."""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before trying again (seconds)
            expected_exception: Exception type that counts as failure
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result or raises exception
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time and 
            time.time() - self.last_failure_time >= self.timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Integration with DeepSentinel
class ResilientSentinelClient:
    """DeepSentinel client with circuit breaker protection."""
    
    def __init__(self, **kwargs):
        """Initialize client with circuit breaker."""
        self.client = SentinelClient(**kwargs)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=30.0,
            expected_exception=Exception
        )
        self.logger = logging.getLogger("resilient_client")
    
    def completion(self, messages, **kwargs):
        """Make completion request through circuit breaker."""
        def _make_request():
            result = safe_completion(self.client, messages, **kwargs)
            
            # Treat non-recoverable errors as circuit breaker failures
            if not result["success"] and not result.get("recoverable", True):
                raise Exception(f"Non-recoverable error: {result['message']}")
            
            return result
        
        try:
            return self.circuit_breaker.call(_make_request)
        except Exception as e:
            self.logger.error(f"Circuit breaker prevented request: {str(e)}")
            return {
                "success": False,
                "error_type": "circuit_breaker",
                "message": f"Circuit breaker is {self.circuit_breaker.state.value}",
                "circuit_state": self.circuit_breaker.state.value
            }

# Usage example
def test_circuit_breaker():
    """Test circuit breaker functionality."""
    client = ResilientSentinelClient(
        sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Simulate multiple failed requests to trigger circuit breaker
    for i in range(6):
        print(f"\nRequest {i+1}:")
        
        # Use invalid model to simulate failures
        result = client.completion(
            [{"role": "user", "content": "Hello"}],
            model="non-existent-model"
        )
        
        print(f"Success: {result['success']}")
        print(f"Error type: {result.get('error_type')}")
        print(f"Circuit state: {result.get('circuit_state', 'N/A')}")
        
        if result.get("circuit_state") == "open":
            print("🔴 Circuit breaker is now OPEN")
            break
        
        time.sleep(1)  # Brief delay between requests

# Run circuit breaker test
# test_circuit_breaker()
```

## Step 8: Complete Error Handling Application

Let's put everything together in a comprehensive error handling application:

```python
import os
import time
import logging
from typing import Dict, List, Optional
from deepsentinel import SentinelClient, ComplianceViolationError

class RobustDeepSentinelApp:
    """Complete application with comprehensive error handling."""
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize the robust application."""
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("robust_app")
        
        # Initialize components
        self.client = SentinelClient(**api_keys)
        self.error_monitor = ErrorMonitor()
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)
        
        # Statistics
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
    
    def chat(self, message: str, context: List[Dict] = None, **kwargs) -> Dict:
        """
        Send a chat message with comprehensive error handling.
        
        Args:
            message: User message
            context: Optional conversation context
            **kwargs: Additional parameters
            
        Returns:
            Dict with response or error information
        """
        self.request_count += 1
        messages = (context or []) + [{"role": "user", "content": message}]
        
        def _make_request():
            return safe_completion(self.client, messages, **kwargs)
        
        try:
            # Make request through circuit breaker
            result = self.circuit_breaker.call(_make_request)
            
            # Record success/failure
            if result["success"]:
                self.success_count += 1
                self.logger.info(f"Request {self.request_count} successful")
            else:
                self.error_count += 1
                self.error_monitor.record_error(result)
                self.logger.warning(f"Request {self.request_count} failed: {result['error_type']}")
            
            return result
            
        except Exception as e:
            # Circuit breaker prevented request
            self.error_count += 1
            error_result = {
                "success": False,
                "error_type": "circuit_breaker",
                "message": f"Circuit breaker prevented request: {str(e)}"
            }
            self.error_monitor.record_error(error_result)
            return error_result
    
    def chat_with_recovery(self, message: str, **kwargs) -> Dict:
        """Chat with automatic recovery for compliance violations."""
        result = self.chat(message, **kwargs)
        
        # If compliance violation, attempt automatic recovery
        if result.get("error_type") == "compliance_violation":
            self.logger.info("Attempting automatic compliance recovery")
            
            # Simple sanitization (remove common patterns)
            sanitized_message = self._sanitize_message(message, result.get("violations", []))
            
            if sanitized_message != message:
                self.logger.info("Retrying with sanitized input")
                recovery_result = self.chat(sanitized_message, **kwargs)
                
                if recovery_result["success"]:
                    recovery_result["recovery_applied"] = True
                    recovery_result["original_message"] = message
                    recovery_result["sanitized_message"] = sanitized_message
                
                return recovery_result
        
        return result
    
    def _sanitize_message(self, message: str, violations: List) -> str:
        """Simple message sanitization."""
        import re
        
        sanitized = message
        
        # Remove common sensitive patterns
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED]', sanitized)  # SSN
        sanitized = re.sub(r'\b(?:\d{4}[- ]){3}\d{4}\b', '[REDACTED]', sanitized)  # Credit card
        sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED]', sanitized)  # Email
        sanitized = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[REDACTED]', sanitized)  # Phone
        
        return sanitized
    
    def get_health_status(self) -> Dict:
        """Get application health status."""
        success_rate = (self.success_count / max(1, self.request_count)) * 100
        
        return {
            "total_requests": self.request_count,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "success_rate": f"{success_rate:.1f}%",
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "error_summary": self.error_monitor.get_error_summary(1)
        }
    
    def demo_conversation(self):
        """Run a demo conversation showing error handling."""
        print("🤖 Robust DeepSentinel Chat Demo")
        print("=" * 50)
        
        test_messages = [
            "Hello! How are you today?",
            "What are the best practices for data security?",
            "My SSN is 123-45-6789 and I need help.",  # Compliance violation
            "Can you explain quantum computing?",
            "Thanks for your help!"
        ]
        
        for i, message in enumerate(test_messages):
            print(f"\n🗣️  User: {message}")
            
            result = self.chat_with_recovery(message)
            
            if result["success"]:
                print(f"🤖 Assistant: {result['content'][:100]}...")
                
                if result.get("recovery_applied"):
                    print(f"⚠️  Note: Input was sanitized due to sensitive content")
            else:
                print(f"❌ Error: {result['message']}")
                
                # Show recovery suggestions
                if result.get("suggested_action"):
                    print(f"💡 Suggestion: {result['suggested_action']}")
        
        # Show final health status
        print("\n" + "="*50)
        print("HEALTH STATUS")
        print("="*50)
        health = self.get_health_status()
        for key, value in health.items():
            if key != "error_summary":
                print(f"{key}: {value}")

# Usage
def main():
    """Run the robust application demo."""
    api_keys = {
        "sentinel_api_key": os.getenv("DEEPSENTINEL_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }
    
    if not all(api_keys.values()):
        print("❌ Please set your API keys in environment variables")
        return
    
    app = RobustDeepSentinelApp(api_keys)
    app.demo_conversation()

if __name__ == "__main__":
    main()
```

## Best Practices Summary

1. **Categorize Errors** - Use specific error types to handle different failure modes appropriately
2. **Implement Retry Logic** - Use exponential backoff for recoverable errors
3. **Provider Failover** - Use multiple providers for resilience
4. **Monitor and Alert** - Track error patterns and set up alerting
5. **Circuit Breakers** - Prevent cascading failures
6. **Graceful Degradation** - Provide fallbacks when possible
7. **User-Friendly Messages** - Transform technical errors into actionable user guidance
8. **Compliance Recovery** - Offer automatic and manual recovery for compliance violations

## What's Next?

Congratulations! You've implemented comprehensive error handling for DeepSentinel. Here are some next steps:

### Advanced Topics
- **[Best Practices](../guides/best-practices.md)** - Production-ready patterns
- **[Performance Optimization](../guides/performance.md)** - Optimize error handling performance
- **[Multiple Providers](multiple-providers.md)** - Advanced provider management

### Production Considerations
- Set up centralized logging and monitoring
- Implement proper alerting systems
- Create error dashboards and reports
- Regular error pattern analysis

## Common Issues

### High Error Rates
**Solution**: Review error patterns, adjust retry strategies, and check provider health

### False Compliance Alerts
**Solution**: Fine-tune compliance policies and detection sensitivity

### Circuit Breaker Stuck Open
**Solution**: Reduce failure threshold or increase timeout period

## Getting Help

- **[FAQ](../faq.md)** - Common questions and solutions
- **[API Reference](../reference/)** - Complete API documentation
- **[GitHub Issues](https://github.com/deepsentinel/deepsentinel-sdk/issues)** - Report bugs

---

**Next Guide**: [Best Practices →](../guides/best-practices.md)