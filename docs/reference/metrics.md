# Metrics

The `deepsentinel.metrics` module provides a comprehensive metrics collection system for monitoring the performance, usage patterns, and operational characteristics of the DeepSentinel SDK.

## Overview

The metrics system is responsible for:

- Collecting performance metrics for LLM requests and responses
- Tracking latency and throughput for provider operations
- Monitoring compliance checks and violations
- Providing data for operational insights and optimization
- Supporting cost analysis and usage patterns

## Components

The metrics system consists of these primary components:

1. **MetricsCollector**: Main interface for collecting and reporting metrics
2. **Metric Types**: Different types of metrics (counters, gauges, histograms)
3. **Reporting Options**: Methods for accessing and exporting collected metrics

## Usage

The metrics system is automatically used by the `SentinelClient` and its components:

```python
import asyncio
from deepsentinel import SentinelClient

async def main():
    client = SentinelClient(
        sentinel_api_key="your-sentinel-api-key",
        openai_api_key="your-openai-api-key",
        # Metrics configuration
        metrics_enabled=True,
        metrics_collection_interval=60,  # seconds
    )
    
    await client.initialize()
    
    # The metrics system automatically collects data during operations
    response = await client.chat.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # Get current metrics
    metrics_data = client.metrics.get_metrics()
    print(f"Total requests: {metrics_data['request_count']}")
    print(f"Average latency: {metrics_data['request_latency_avg']} ms")
    
    await client.cleanup()

asyncio.run(main())
```

## MetricsCollector

The `MetricsCollector` class provides the main interface for metrics operations.

```python
class MetricsCollector:
    def __init__(
        self,
        config: Union[Dict[str, Any], "MetricsConfig"]
    ) -> None
```

**Parameters:**

- **config** (`Union[Dict[str, Any], MetricsConfig]`): Metrics configuration

### Methods

#### `record_request`

```python
def record_request(
    self,
    provider: str,
    operation: str,
    model: str,
    request_id: str,
    start_time: float
) -> None
```

Record the start of an LLM provider request.

**Parameters:**

- **provider** (`str`): Provider name
- **operation** (`str`): Operation type (e.g., "chat.create")
- **model** (`str`): Model name
- **request_id** (`str`): Unique identifier for the request
- **start_time** (`float`): Request start time (timestamp)

#### `record_response`

```python
def record_response(
    self,
    request_id: str,
    status: str,
    tokens: Optional[Dict[str, int]] = None,
    end_time: Optional[float] = None
) -> None
```

Record the completion of an LLM provider request.

**Parameters:**

- **request_id** (`str`): Unique identifier for the request
- **status** (`str`): Response status ("success", "error")
- **tokens** (`Dict[str, int]`, optional): Token usage information
- **end_time** (`float`, optional): Request end time (timestamp)

#### `record_compliance_check`

```python
def record_compliance_check(
    self,
    policy_name: str,
    passed: bool,
    latency_ms: float
) -> None
```

Record a compliance policy check.

**Parameters:**

- **policy_name** (`str`): Name of the policy
- **passed** (`bool`): Whether the check passed
- **latency_ms** (`float`): Time taken for the check in milliseconds

#### `record_error`

```python
def record_error(
    self,
    error_type: str,
    provider: Optional[str] = None,
    operation: Optional[str] = None
) -> None
```

Record an error event.

**Parameters:**

- **error_type** (`str`): Type of error
- **provider** (`str`, optional): Provider name
- **operation** (`str`, optional): Operation type

#### `increment_counter`

```python
def increment_counter(
    self,
    name: str,
    value: int = 1,
    labels: Optional[Dict[str, str]] = None
) -> None
```

Increment a counter metric.

**Parameters:**

- **name** (`str`): Metric name
- **value** (`int`, optional): Increment value
- **labels** (`Dict[str, str]`, optional): Metric labels

#### `record_gauge`

```python
def record_gauge(
    self,
    name: str,
    value: float,
    labels: Optional[Dict[str, str]] = None
) -> None
```

Record a gauge metric.

**Parameters:**

- **name** (`str`): Metric name
- **value** (`float`): Gauge value
- **labels** (`Dict[str, str]`, optional): Metric labels

#### `record_histogram`

```python
def record_histogram(
    self,
    name: str,
    value: float,
    labels: Optional[Dict[str, str]] = None
) -> None
```

Record a value for a histogram metric.

**Parameters:**

- **name** (`str`): Metric name
- **value** (`float`): Observed value
- **labels** (`Dict[str, str]`, optional): Metric labels

#### `get_metrics`

```python
def get_metrics(self) -> Dict[str, Any]
```

Get the current metrics data.

**Returns:**

- `Dict[str, Any]`: Dictionary containing all metrics

#### `export_metrics`

```python
def export_metrics(
    self,
    format: str = "json"
) -> Union[str, Dict[str, Any]]
```

Export metrics in the specified format.

**Parameters:**

- **format** (`str`): Export format ("json", "prometheus")

**Returns:**

- `Union[str, Dict[str, Any]]`: Metrics data in the specified format

#### `reset_metrics`

```python
def reset_metrics(self) -> None
```

Reset all metrics to their initial values.

## Collected Metrics

The metrics system collects various metrics by default:

### Request/Response Metrics

- **request_count**: Total number of requests (counter)
- **request_latency**: Request latency in milliseconds (histogram)
- **request_latency_avg**: Average request latency (gauge)
- **request_error_count**: Number of request errors (counter)
- **tokens_total**: Total tokens used (counter)
- **prompt_tokens**: Prompt tokens used (counter)
- **completion_tokens**: Completion tokens used (counter)
- **stream_chunks**: Number of streaming chunks received (counter)

### Provider Metrics

- **provider_request_count**: Requests per provider (counter)
- **provider_error_count**: Errors per provider (counter)
- **provider_latency**: Latency per provider (histogram)
- **model_usage**: Requests per model (counter)

### Compliance Metrics

- **compliance_check_count**: Total compliance checks performed (counter)
- **compliance_violation_count**: Number of compliance violations (counter)
- **compliance_check_latency**: Compliance check latency (histogram)
- **redaction_count**: Number of redactions performed (counter)

### System Metrics

- **active_requests**: Currently active requests (gauge)
- **memory_usage**: SDK memory usage in MB (gauge)
- **uptime_seconds**: SDK uptime in seconds (gauge)

## Configuration

The metrics system can be configured through the `SentinelConfig` class:

```python
client = SentinelClient(
    # ... other configuration ...
    
    # Metrics configuration
    metrics_enabled=True,  # Enable/disable metrics collection
    metrics_collection_interval=60,  # Collection interval in seconds
    metrics_export_enabled=False,  # Enable automatic export
    metrics_export_url="",  # URL for metrics export
)
```

### Configuration Options

- **metrics_enabled** (`bool`): Enable or disable metrics collection
- **metrics_collection_interval** (`int`): How often to collect system metrics in seconds
- **metrics_export_enabled** (`bool`): Enable automatic export of metrics
- **metrics_export_url** (`str`): URL endpoint for metrics export
- **metrics_export_interval** (`int`): How often to export metrics in seconds
- **metrics_export_format** (`str`): Export format ("json", "prometheus")
- **metrics_detailed_providers** (`bool`): Collect detailed provider metrics
- **metrics_detailed_models** (`bool`): Collect detailed model-level metrics

## Examples

### Basic Metrics Usage

```python
import asyncio
from deepsentinel import SentinelClient

async def main():
    client = SentinelClient(
        sentinel_api_key="your-sentinel-api-key",
        openai_api_key="your-openai-api-key",
        metrics_enabled=True
    )
    
    await client.initialize()
    
    # Run multiple requests
    for i in range(5):
        await client.chat.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Question {i}: What is AI?"}]
        )
    
    # Get metrics
    metrics = client.metrics.get_metrics()
    
    # Print request statistics
    print(f"Total requests: {metrics['request_count']}")
    print(f"Average latency: {metrics['request_latency_avg']:.2f} ms")
    print(f"Total tokens: {metrics['tokens_total']}")
    
    # Print compliance metrics
    print(f"Compliance checks: {metrics['compliance_check_count']}")
    print(f"Compliance violations: {metrics['compliance_violation_count']}")
    
    await client.cleanup()

asyncio.run(main())
```

### Performance Monitoring

```python
import asyncio
import time
from deepsentinel import SentinelClient

async def monitor_performance():
    client = SentinelClient(
        sentinel_api_key="your-sentinel-api-key",
        openai_api_key="your-openai-api-key",
        anthropic_api_key="your-anthropic-api-key",
        metrics_enabled=True
    )
    
    await client.initialize()
    
    # Monitor performance across different models and providers
    models = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-opus-20240229"
    }
    
    for provider, model in models.items():
        print(f"Testing {provider} model: {model}...")
        
        start_time = time.time()
        response = await client.chat.create(
            model=model,
            messages=[{
                "role": "user", 
                "content": "Explain quantum computing in simple terms."
            }],
            provider=provider
        )
        duration = time.time() - start_time
        
        print(f"  Response time: {duration:.2f} seconds")
        print(f"  Response length: {len(response.choices[0].message.content)} chars")
        if response.usage:
            print(f"  Tokens: {response.usage.total_tokens} total")
    
    # Get overall metrics
    metrics = client.metrics.get_metrics()
    
    # Compare provider performance
    for provider in models.keys():
        provider_latency = metrics.get(f"provider_{provider}_latency_avg", 0)
        print(f"{provider} average latency: {provider_latency:.2f} ms")
    
    await client.cleanup()

asyncio.run(monitor_performance())
```

### Custom Metrics

```python
import asyncio
import random
from deepsentinel import SentinelClient

async def track_custom_metrics():
    client = SentinelClient(
        sentinel_api_key="your-sentinel-api-key",
        openai_api_key="your-openai-api-key",
        metrics_enabled=True
    )
    
    await client.initialize()
    
    # Define our application component
    component = "recommendation_engine"
    
    # Record custom metrics
    for i in range(10):
        # Simulate some application-specific metric
        process_time = random.uniform(10, 100)
        client.metrics.record_histogram(
            name="app_process_time",
            value=process_time,
            labels={"component": component, "task": "user_recommendations"}
        )
        
        # Increment a custom counter
        client.metrics.increment_counter(
            name="recommendations_generated",
            value=5,
            labels={"component": component, "user_segment": "premium"}
        )
        
        # Record a gauge value
        client.metrics.record_gauge(
            name="recommendation_quality",
            value=random.uniform(0.7, 0.99),
            labels={"component": component}
        )
    
    # Get all metrics including our custom ones
    metrics = client.metrics.get_metrics()
    
    # Find our custom metrics
    custom_metrics = {k: v for k, v in metrics.items() 
                     if k.startswith("app_") or 
                        k.startswith("recommendation")}
    
    print("Custom metrics:")
    for name, value in custom_metrics.items():
        print(f"  {name}: {value}")
    
    await client.cleanup()

asyncio.run(track_custom_metrics())
```

### Exporting Metrics to External Systems

```python
import asyncio
import requests
from deepsentinel import SentinelClient

async def export_metrics_example():
    client = SentinelClient(
        sentinel_api_key="your-sentinel-api-key",
        openai_api_key="your-openai-api-key",
        metrics_enabled=True,
        # Auto export is disabled, we'll do it manually
        metrics_export_enabled=False
    )
    
    await client.initialize()
    
    # Make some requests to generate metrics
    for prompt in ["Tell me about AI", "Explain neural networks", "What is deep learning?"]:
        await client.chat.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
    
    # Manually export metrics to a monitoring system
    metrics_json = client.metrics.export_metrics(format="json")
    
    # Export to a hypothetical monitoring system (commented out)
    """
    response = requests.post(
        "https://monitoring.example.com/api/metrics",
        json=metrics_json,
        headers={"Authorization": "Bearer your-monitoring-token"}
    )
    print(f"Metrics export status: {response.status_code}")
    """
    
    # Export as Prometheus format
    prometheus_metrics = client.metrics.export_metrics(format="prometheus")
    print("Prometheus format metrics sample:")
    print(prometheus_metrics[:500] + "...")  # Print just a sample
    
    await client.cleanup()

asyncio.run(export_metrics_example())
```

## Best Practices

1. **Enable metrics in production:** Metrics provide valuable insights into your application's performance and behavior.

2. **Monitor key indicators:** Pay attention to latency, token usage, and compliance violations.

3. **Set up alerts:** Configure alerts for unexpected changes in error rates or latency.

4. **Analyze patterns over time:** Look for trends in usage patterns to optimize your application.

5. **Use custom metrics:** Add application-specific metrics to track business-level indicators.

6. **Export to monitoring systems:** Integrate with external monitoring tools for comprehensive dashboards.

7. **Periodically reset metrics:** For long-running applications, consider resetting metrics periodically to avoid memory growth.

8. **Compare providers:** Use metrics to compare performance and cost across different providers and models.
