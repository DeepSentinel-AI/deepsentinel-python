# Best Practices Guide

This guide covers recommended patterns and practices for using DeepSentinel in production applications. Following these best practices will help you build secure, performant, and maintainable applications.

## Overview

This guide covers:
- Security and API key management
- Configuration management across environments
- Error handling and resilience patterns
- Logging and monitoring strategies
- Testing and validation approaches
- Performance optimization techniques

**Time to read:** 20 minutes

## Security Best Practices

### API Key Management

**✅ DO: Use Environment Variables**
```python
import os
from deepsentinel import SentinelClient

# Good: Load from environment variables
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)
```

**❌ DON'T: Hardcode API Keys**
```python
# Bad: Never hardcode API keys
client = SentinelClient(
    sentinel_api_key="sk-deepsentinel-abc123...",  # DON'T DO THIS
    openai_api_key="sk-abc123..."  # DON'T DO THIS
)
```

**✅ DO: Use Secret Management Services**
```python
import boto3
from deepsentinel import SentinelClient

def get_secret(secret_name, region_name="us-east-1"):
    """Retrieve secret from AWS Secrets Manager."""
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Load secrets from AWS Secrets Manager
secrets = get_secret("deepsentinel/api-keys")

client = SentinelClient(
    sentinel_api_key=secrets["sentinel_api_key"],
    openai_api_key=secrets["openai_api_key"]
)
```

### Input Sanitization

**✅ DO: Validate Input Before Processing**
```python
import re
from typing import List, Dict

def validate_user_input(message: str) -> tuple[bool, str]:
    """
    Validate user input before sending to LLM.
    
    Returns:
        (is_valid, error_message)
    """
    # Check length
    if len(message) > 10000:
        return False, "Message too long (max 10,000 characters)"
    
    # Check for obvious injection attempts
    suspicious_patterns = [
        r'<script.*?>',
        r'javascript:',
        r'eval\s*\(',
        r'exec\s*\('
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return False, "Message contains potentially harmful content"
    
    return True, ""

def safe_chat(client, user_message: str, **kwargs):
    """Make a chat request with input validation."""
    # Validate input
    is_valid, error_msg = validate_user_input(user_message)
    if not is_valid:
        return {"success": False, "error": error_msg}
    
    # Proceed with request
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": user_message}],
            **kwargs
        )
        return {"success": True, "content": response.choices[0].message.content}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### Compliance Policy Management

**✅ DO: Use Environment-Specific Policies**
```python
from deepsentinel import CompliancePolicy, SentinelConfig

def create_compliance_policies(environment: str) -> List[CompliancePolicy]:
    """Create policies based on deployment environment."""
    
    if environment == "production":
        # Strict policies for production
        return [
            CompliancePolicy(
                name="prod-policy",
                enable_pii_detection=True,
                enable_phi_detection=True,
                enable_pci_detection=True,
                max_risk_score=0.6,  # Strict threshold
                block_on_violation=True,
                log_violations=True,
                anonymize_audit_logs=True
            )
        ]
    
    elif environment == "staging":
        # Moderate policies for staging
        return [
            CompliancePolicy(
                name="staging-policy",
                enable_pii_detection=True,
                enable_phi_detection=True,
                max_risk_score=0.8,
                block_on_violation=False,  # Allow for testing
                anonymize_sensitive_data=True,
                log_violations=True
            )
        ]
    
    else:  # development
        # Lenient policies for development
        return [
            CompliancePolicy(
                name="dev-policy",
                enable_pii_detection=True,
                max_risk_score=0.9,
                block_on_violation=False,
                log_violations=False  # Reduce noise in dev
            )
        ]

# Usage
environment = os.getenv("ENVIRONMENT", "development")
policies = create_compliance_policies(environment)

config = SentinelConfig(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    compliance_policies=policies
)
```

## Configuration Management

### Environment-Based Configuration

**✅ DO: Use Configuration Classes**
```python
from dataclasses import dataclass
from typing import List, Optional
import os

@dataclass
class DeepSentinelConfig:
    """Configuration for DeepSentinel client."""
    
    # API Keys
    sentinel_api_key: str
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Environment settings
    environment: str = "development"
    debug_mode: bool = False
    
    # Compliance settings
    compliance_enabled: bool = True
    max_risk_score: float = 0.8
    
    # Performance settings
    timeout: float = 30.0
    max_retries: int = 3
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_environment(cls) -> 'DeepSentinelConfig':
        """Create configuration from environment variables."""
        return cls(
            sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            environment=os.getenv("ENVIRONMENT", "development"),
            debug_mode=os.getenv("DEBUG", "false").lower() == "true",
            compliance_enabled=os.getenv("COMPLIANCE_ENABLED", "true").lower() == "true",
            max_risk_score=float(os.getenv("MAX_RISK_SCORE", "0.8")),
            timeout=float(os.getenv("REQUEST_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE")
        )

# Usage
config = DeepSentinelConfig.from_environment()

# Validate required fields
if not config.sentinel_api_key:
    raise ValueError("DEEPSENTINEL_API_KEY environment variable is required")

# Create client with configuration
client = SentinelClient(
    sentinel_api_key=config.sentinel_api_key,
    openai_api_key=config.openai_api_key,
    anthropic_api_key=config.anthropic_api_key,
    debug_mode=config.debug_mode
)
```

### Configuration Validation

**✅ DO: Validate Configuration on Startup**
```python
def validate_configuration(config: DeepSentinelConfig) -> List[str]:
    """Validate configuration and return list of errors."""
    errors = []
    
    # Required fields
    if not config.sentinel_api_key:
        errors.append("DeepSentinel API key is required")
    
    if not config.openai_api_key and not config.anthropic_api_key:
        errors.append("At least one LLM provider API key is required")
    
    # Value ranges
    if not 0.0 <= config.max_risk_score <= 1.0:
        errors.append("max_risk_score must be between 0.0 and 1.0")
    
    if config.timeout <= 0:
        errors.append("timeout must be positive")
    
    if config.max_retries < 0:
        errors.append("max_retries must be non-negative")
    
    # Environment-specific validation
    if config.environment == "production":
        if config.debug_mode:
            errors.append("debug_mode should be disabled in production")
        
        if config.max_risk_score > 0.8:
            errors.append("max_risk_score should be stricter in production")
    
    return errors

# Validate configuration
config = DeepSentinelConfig.from_environment()
validation_errors = validate_configuration(config)

if validation_errors:
    print("Configuration errors:")
    for error in validation_errors:
        print(f"  - {error}")
    sys.exit(1)
```

## Error Handling Patterns

### Comprehensive Error Handling

**✅ DO: Handle All Error Types**
```python
from deepsentinel import (
    ComplianceViolationError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    NetworkError
)
import logging
import time

logger = logging.getLogger(__name__)

def robust_completion(client, messages, max_retries=3, **kwargs):
    """Make completion with comprehensive error handling."""
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                messages=messages,
                **kwargs
            )
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model": response.model,
                "tokens": response.usage.total_tokens
            }
        
        except ComplianceViolationError as e:
            # Don't retry compliance violations
            logger.warning(f"Compliance violation: {e.message}")
            return {
                "success": False,
                "error_type": "compliance",
                "message": e.message,
                "violations": [v.type for v in e.violations],
                "retryable": False
            }
        
        except AuthenticationError as e:
            # Don't retry authentication errors
            logger.error(f"Authentication error: {e.message}")
            return {
                "success": False,
                "error_type": "authentication",
                "message": e.message,
                "retryable": False
            }
        
        except RateLimitError as e:
            # Retry rate limit errors with backoff
            if attempt < max_retries:
                wait_time = e.retry_after or (2 ** attempt)
                logger.info(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)
                continue
            
            return {
                "success": False,
                "error_type": "rate_limit",
                "message": e.message,
                "retryable": True
            }
        
        except (ProviderError, NetworkError) as e:
            # Retry provider and network errors
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Provider/network error, retrying in {wait_time}s")
                time.sleep(wait_time)
                continue
            
            return {
                "success": False,
                "error_type": "provider" if isinstance(e, ProviderError) else "network",
                "message": str(e),
                "retryable": True
            }
        
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "error_type": "unexpected",
                "message": str(e),
                "retryable": False
            }
    
    return {
        "success": False,
        "error_type": "max_retries_exceeded",
        "message": f"Failed after {max_retries} retries",
        "retryable": False
    }
```

### Graceful Degradation

**✅ DO: Provide Fallbacks**
```python
def chat_with_fallback(client, user_message: str, **kwargs):
    """Chat with graceful fallback options."""
    
    # Primary attempt
    result = robust_completion(
        client, 
        [{"role": "user", "content": user_message}], 
        **kwargs
    )
    
    if result["success"]:
        return result
    
    # Fallback 1: Try with different provider
    if "provider" in kwargs:
        fallback_providers = {"openai": "anthropic", "anthropic": "openai"}
        fallback_provider = fallback_providers.get(kwargs["provider"])
        
        if fallback_provider:
            logger.info(f"Trying fallback provider: {fallback_provider}")
            kwargs["provider"] = fallback_provider
            
            fallback_result = robust_completion(
                client,
                [{"role": "user", "content": user_message}],
                **kwargs
            )
            
            if fallback_result["success"]:
                fallback_result["used_fallback"] = True
                return fallback_result
    
    # Fallback 2: Return helpful error message
    if result["error_type"] == "compliance":
        return {
            "success": False,
            "fallback_message": "I can't process that request because it contains sensitive information. Please try rephrasing without personal details.",
            "original_error": result
        }
    
    elif result["error_type"] == "rate_limit":
        return {
            "success": False,
            "fallback_message": "I'm currently experiencing high demand. Please try again in a few moments.",
            "original_error": result
        }
    
    else:
        return {
            "success": False,
            "fallback_message": "I'm having trouble processing your request right now. Please try again later.",
            "original_error": result
        }
```

## Logging and Monitoring

### Structured Logging

**✅ DO: Use Structured Logging**
```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    """Structured logger for DeepSentinel applications."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if os.getenv("LOG_FILE"):
            file_handler = logging.FileHandler(os.getenv("LOG_FILE"))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_request(self, event_type: str, data: Dict[str, Any], level: str = "info"):
        """Log structured request data."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        getattr(self.logger, level)(json.dumps(log_entry))
    
    def log_completion(self, messages: list, response: dict, duration: float):
        """Log completion request and response."""
        self.log_request("completion", {
            "message_count": len(messages),
            "model": response.get("model"),
            "tokens": response.get("tokens", 0),
            "duration_ms": int(duration * 1000),
            "success": response.get("success", False)
        })
    
    def log_error(self, error_type: str, error_message: str, context: Dict = None):
        """Log error with context."""
        self.log_request("error", {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }, level="error")

# Usage
logger = StructuredLogger("deepsentinel_app")

def logged_completion(client, messages, **kwargs):
    """Make completion with structured logging."""
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(messages=messages, **kwargs)
        duration = time.time() - start_time
        
        result = {
            "success": True,
            "content": response.choices[0].message.content,
            "model": response.model,
            "tokens": response.usage.total_tokens
        }
        
        logger.log_completion(messages, result, duration)
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        
        logger.log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            context={
                "message_count": len(messages),
                "duration_ms": int(duration * 1000),
                "kwargs": kwargs
            }
        )
        raise
```

### Metrics Collection

**✅ DO: Collect Key Metrics**
```python
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import threading

class MetricsCollector:
    """Collect and track application metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(int)
        self.timings = defaultdict(list)
        self.errors = Counter()
        self.lock = threading.Lock()
        
    def record_request(self, duration: float, success: bool, model: str = None):
        """Record request metrics."""
        with self.lock:
            self.metrics["total_requests"] += 1
            if success:
                self.metrics["successful_requests"] += 1
            else:
                self.metrics["failed_requests"] += 1
            
            self.timings["request_duration"].append(duration)
            
            if model:
                self.metrics[f"requests_{model}"] += 1
    
    def record_error(self, error_type: str):
        """Record error metrics."""
        with self.lock:
            self.errors[error_type] += 1
    
    def record_tokens(self, token_count: int, model: str = None):
        """Record token usage."""
        with self.lock:
            self.metrics["total_tokens"] += token_count
            if model:
                self.metrics[f"tokens_{model}"] += token_count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self.lock:
            total_requests = self.metrics["total_requests"]
            if total_requests == 0:
                return {"message": "No requests recorded"}
            
            durations = self.timings["request_duration"]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            return {
                "total_requests": total_requests,
                "successful_requests": self.metrics["successful_requests"],
                "failed_requests": self.metrics["failed_requests"],
                "success_rate": (self.metrics["successful_requests"] / total_requests) * 100,
                "average_duration_ms": int(avg_duration * 1000),
                "total_tokens": self.metrics["total_tokens"],
                "error_breakdown": dict(self.errors),
                "requests_per_model": {
                    k.replace("requests_", ""): v 
                    for k, v in self.metrics.items() 
                    if k.startswith("requests_") and k != "requests"
                }
            }

# Global metrics collector
metrics = MetricsCollector()

def instrumented_completion(client, messages, **kwargs):
    """Make completion with metrics collection."""
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(messages=messages, **kwargs)
        duration = time.time() - start_time
        
        metrics.record_request(duration, True, response.model)
        metrics.record_tokens(response.usage.total_tokens, response.model)
        
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "model": response.model,
            "tokens": response.usage.total_tokens
        }
        
    except Exception as e:
        duration = time.time() - start_time
        metrics.record_request(duration, False)
        metrics.record_error(type(e).__name__)
        raise
```

## Testing Strategies

### Unit Testing

**✅ DO: Test Your Integration Code**
```python
import unittest
from unittest.mock import Mock, patch
from deepsentinel import ComplianceViolationError

class TestDeepSentinelIntegration(unittest.TestCase):
    """Test DeepSentinel integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.test_messages = [{"role": "user", "content": "Hello"}]
    
    def test_successful_completion(self):
        """Test successful completion."""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello there!"
        mock_response.model = "gpt-4o"
        mock_response.usage.total_tokens = 10
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Test
        result = safe_completion(self.mock_client, self.test_messages)
        
        # Assertions
        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "Hello there!")
        self.assertEqual(result["model"], "gpt-4o")
        self.assertEqual(result["tokens"], 10)
    
    def test_compliance_violation(self):
        """Test compliance violation handling."""
        # Mock compliance violation
        violation = Mock()
        violation.type = "ssn"
        
        error = ComplianceViolationError(
            message="SSN detected",
            violations=[violation],
            risk_score=0.9
        )
        
        self.mock_client.chat.completions.create.side_effect = error
        
        # Test
        result = safe_completion(self.mock_client, self.test_messages)
        
        # Assertions
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "compliance_violation")
        self.assertEqual(result["violations"], ["ssn"])
        self.assertEqual(result["risk_score"], 0.9)
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_retry_logic(self, mock_sleep):
        """Test retry logic for recoverable errors."""
        # Mock rate limit error then success
        rate_limit_error = RateLimitError(
            message="Rate limited",
            retry_after=1
        )
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Success!"
        
        self.mock_client.chat.completions.create.side_effect = [
            rate_limit_error,  # First call fails
            mock_response      # Second call succeeds
        ]
        
        # Test
        result = robust_completion(self.mock_client, self.test_messages, max_retries=1)
        
        # Assertions
        self.assertTrue(result["success"])
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 2)
        mock_sleep.assert_called_once()

if __name__ == "__main__":
    unittest.main()
```

### Integration Testing

**✅ DO: Test with Real API (Sparingly)**
```python
import pytest
import os
from deepsentinel import SentinelClient

@pytest.mark.integration
class TestDeepSentinelIntegration:
    """Integration tests for DeepSentinel."""
    
    @pytest.fixture
    def client(self):
        """Create client for testing."""
        api_key = os.getenv("DEEPSENTINEL_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key or not openai_key:
            pytest.skip("API keys not available")
        
        return SentinelClient(
            sentinel_api_key=api_key,
            openai_api_key=openai_key
        )
    
    def test_basic_completion(self, client):
        """Test basic completion works."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            max_tokens=10
        )
        
        assert response.choices[0].message.content
        assert "test" in response.choices[0].message.content.lower()
    
    def test_compliance_detection(self, client):
        """Test that compliance detection works."""
        with pytest.raises(ComplianceViolationError):
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "My SSN is 123-45-6789"}]
            )

# Run with: pytest test_integration.py -m integration
```

## Performance Optimization

### Connection Pooling

**✅ DO: Configure Connection Pooling**
```python
from deepsentinel import SentinelClient, SentinelConfig, PerformanceConfig

# Configure performance settings
perf_config = PerformanceConfig(
    connection_pool_size=20,        # Total connections
    connection_pool_per_host=5,     # Per provider
    connect_timeout=5.0,            # Connection timeout
    read_timeout=30.0,              # Read timeout
    keep_alive=True,                # Keep connections alive
    connection_pool_ttl=300         # Pool TTL in seconds
)

config = SentinelConfig(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    performance_config=perf_config
)

client = SentinelClient(config=config)
```

### Caching Strategies

**✅ DO: Implement Response Caching**
```python
import hashlib
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

class ResponseCache:
    """Simple in-memory response cache."""
    
    def __init__(self, ttl_minutes: int = 60, max_size: int = 1000):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_size = max_size
    
    def _generate_key(self, messages: list, model: str, **kwargs) -> str:
        """Generate cache key from request parameters."""
        cache_data = {
            "messages": messages,
            "model": model,
            **{k: v for k, v in kwargs.items() if k in ["temperature", "max_tokens"]}
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()[:16]
    
    def get(self, messages: list, model: str, **kwargs) -> Optional[Dict]:
        """Get cached response if available and not expired."""
        key = self._generate_key(messages, model, **kwargs)
        
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry["expires"]:
                return entry["response"]
            else:
                del self.cache[key]  # Remove expired entry
        
        return None
    
    def set(self, messages: list, model: str, response: Dict, **kwargs):
        """Cache response."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["created"])
            del self.cache[oldest_key]
        
        key = self._generate_key(messages, model, **kwargs)
        self.cache[key] = {
            "response": response,
            "created": datetime.now(),
            "expires": datetime.now() + self.ttl
        }

# Global cache instance
response_cache = ResponseCache(ttl_minutes=30, max_size=500)

def cached_completion(client, messages, **kwargs):
    """Make completion with response caching."""
    model = kwargs.get("model", "gpt-4o")
    
    # Check cache first
    cached_response = response_cache.get(messages, model, **kwargs)
    if cached_response:
        cached_response["from_cache"] = True
        return cached_response
    
    # Make request
    result = robust_completion(client, messages, **kwargs)
    
    # Cache successful responses
    if result["success"]:
        response_cache.set(messages, model, result, **kwargs)
    
    return result
```

## Production Deployment

### Health Checks

**✅ DO: Implement Health Checks**
```python
from datetime import datetime, timedelta
import asyncio

class HealthChecker:
    """Health check implementation for DeepSentinel apps."""
    
    def __init__(self, client: SentinelClient):
        self.client = client
        self.last_check = None
        self.last_status = None
        self.check_interval = timedelta(minutes=5)
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "checks": {}
        }
        
        # Check 1: API connectivity
        try:
            test_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "health check"}],
                max_tokens=1
            )
            health_status["checks"]["api_connectivity"] = {
                "status": "pass",
                "response_time_ms": 100  # You'd measure this
            }
        except Exception as e:
            health_status["checks"]["api_connectivity"] = {
                "status": "fail",
                "error": str(e)
            }
            health_status["status"] = "unhealthy"
        
        # Check 2: Compliance engine
        try:
            # Test compliance detection with known violation
            self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "SSN: 123-45-6789"}],
                max_tokens=1
            )
            # If we get here, compliance didn't catch it - that's bad
            health_status["checks"]["compliance"] = {
                "status": "fail",
                "error": "Compliance detection not working"
            }
            health_status["status"] = "unhealthy"
        except ComplianceViolationError:
            # Good - compliance caught the violation
            health_status["checks"]["compliance"] = {
                "status": "pass"
            }
        except Exception as e:
            health_status["checks"]["compliance"] = {
                "status": "fail",
                "error": str(e)
            }
            health_status["status"] = "unhealthy"
        
        self.last_check = datetime.utcnow()
        self.last_status = health_status
        return health_status
    
    def get_cached_status(self) -> Dict[str, Any]:
        """Get cached health status if recent enough."""
        if (self.last_check and self.last_status and 
            datetime.utcnow() - self.last_check < self.check_interval):
            return self.last_status
        
        # Return stale status with warning
        if self.last_status:
            stale_status = self.last_status.copy()
            stale_status["warning"] = "Health check data is stale"
            return stale_status
        
        return {"status": "unknown", "error": "No health check data available"}

# Usage in web framework (e.g., FastAPI)
from fastapi import FastAPI

app = FastAPI()
health_checker = HealthChecker(client)

@app.get("/health")
async def health_endpoint():
    """Health check endpoint."""
    return await health_checker.check_health()

@app.get("/health/quick")
async def quick_health_endpoint():
    """Quick health check using cached data."""
    return health_checker.get_cached_status()
```

### Monitoring Integration

**✅ DO: Integrate with Monitoring Systems**
```python
import requests
import json
from datetime import datetime

class MonitoringIntegration:
    """Integration with external monitoring systems."""
    
    def __init__(self, webhook_url: str = None, service_name: str = "deepsentinel-app"):
        self.webhook_url = webhook_url
        self.service_name = service_name
    
    def send_metrics(self, metrics: Dict[str, Any]):
        """Send metrics to monitoring system."""
        if not self.webhook_url:
            return
        
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "metrics": metrics
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send metrics: {e}")
    
    def send_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Send alert to monitoring system."""
        if not self.webhook_url:
            return
        
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "alert_type": alert_type,
            "message": message,
            "severity": severity
        }
        
        try:
            response = requests.post(
                f"{self.webhook_url}/alerts",
                json=payload,
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

# Usage
monitoring = MonitoringIntegration(
    webhook_url=os.getenv("MONITORING_WEBHOOK_URL"),
    service_name="my-deepsentinel-app"
)

# Send periodic metrics
def send_periodic_metrics():
    """Send metrics to monitoring system."""
    summary = metrics.get_summary()
    monitoring.send_metrics(summary)
    
    # Send alerts for concerning metrics
    if summary.get("success_rate", 100) < 95:
        monitoring.send_alert(
            "low_success_rate",
            f"Success rate dropped to {summary['success_rate']:.1f}%",
            "warning"
        )

# Schedule periodic metrics sending
import threading
import time

def metrics_sender():
    """Background thread to send metrics."""
    while True:
        try:
            send_periodic_metrics()
        except Exception as e:
            logger.error(f"Error sending metrics: {e}")
        time.sleep(300)  # Send every 5 minutes

# Start metrics sender in background
metrics_thread = threading.Thread(target=metrics_sender, daemon=True)
metrics_thread.start()
```

## Summary Checklist

### Security ✅
- [ ] API keys stored in environment variables or secret management
- [ ] Input validation implemented
- [ ] Environment-specific compliance policies
- [ ] Audit logging enabled for production

### Configuration ✅
- [ ] Configuration classes for different environments
- [ ] Configuration validation on startup
- [ ] Secrets properly managed
- [ ] Debug mode disabled in production

### Error Handling ✅
- [ ] All error types handled appropriately
- [ ] Retry logic with exponential backoff
- [ ] Graceful degradation and fallbacks
- [ ] Circuit breaker for external dependencies

### Monitoring ✅
- [ ] Structured logging implemented
- [ ] Key metrics collected and tracked
- [ ] Health checks configured
- [ ] Alerting set up for critical issues

### Testing ✅
- [ ] Unit tests for integration code
- [ ] Integration tests (run sparingly)
- [ ] Mock external dependencies
- [ ] Test error scenarios

### Performance ✅
- [ ] Connection pooling configured
- [ ] Response caching where appropriate
- [ ] Timeout values optimized
- [ ] Resource cleanup implemented

### Production ✅
- [ ] Health check endpoints
- [ ] Monitoring integration
- [ ] Deployment automation
- [ ] Documentation updated

Following these best practices will help ensure your DeepSentinel integration is secure, reliable, and maintainable in production environments.

---

**Next Guide**: [Performance Optimization →](performance.md)