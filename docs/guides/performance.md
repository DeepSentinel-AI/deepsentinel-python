# Performance Optimization Guide

This guide covers techniques for optimizing DeepSentinel performance in high-throughput applications while maintaining compliance coverage and minimizing latency impact.

## Overview

This guide covers:
- Local vs cloud detection strategies
- Caching optimization techniques
- Request batching and connection pooling
- Performance monitoring and tuning
- Scalability considerations

**Time to read:** 25 minutes

## Understanding Performance Impact

DeepSentinel adds several layers of processing to your LLM requests:

1. **Compliance Detection** - Scanning for sensitive data patterns
2. **Policy Evaluation** - Applying compliance rules
3. **Network Overhead** - Additional API calls to DeepSentinel service
4. **Audit Logging** - Recording compliance events

Let's optimize each layer systematically.

## Compliance Detection Optimization

### Local vs Cloud Detection

**Local Detection (Fastest)**
```python
from deepsentinel import SentinelClient, SentinelConfig, DetectionConfig

# Configure local-first detection
detection_config = DetectionConfig(
    mode="local_first",              # Try local detection first
    local_patterns_only=True,        # Use only local regex patterns
    cloud_fallback=False,            # Disable cloud fallback for speed
    pattern_cache_size=1000,         # Cache compiled patterns
    enable_pattern_optimization=True  # Optimize pattern compilation
)

config = SentinelConfig(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    detection_config=detection_config
)

client = SentinelClient(config=config)
```

**Hybrid Detection (Balanced)**
```python
# Configure hybrid detection for balance of speed and accuracy
detection_config = DetectionConfig(
    mode="hybrid",                   # Local first, cloud for complex cases
    local_confidence_threshold=0.8,  # Use cloud if local confidence < 0.8
    cloud_timeout=500,               # 500ms timeout for cloud detection
    cache_cloud_results=True,        # Cache cloud detection results
    cloud_cache_ttl=3600            # Cache cloud results for 1 hour
)
```

**Cloud Detection (Most Accurate)**
```python
# Configure cloud detection for maximum accuracy
detection_config = DetectionConfig(
    mode="cloud_first",              # Always use cloud detection
    cloud_timeout=2000,              # 2 second timeout
    local_fallback=True,             # Fall back to local if cloud fails
    enable_ml_detection=True         # Enable ML-based detection
)
```

### Pattern Optimization

**Optimize Regular Expressions**
```python
import re
from deepsentinel import PatternOptimizer

class OptimizedPatternMatcher:
    """Optimized pattern matcher for high-performance detection."""
    
    def __init__(self):
        self.compiled_patterns = {}
        self.pattern_cache = {}
        self.optimizer = PatternOptimizer()
    
    def add_patterns(self, patterns: dict):
        """Add and optimize patterns."""
        for name, pattern in patterns.items():
            # Optimize pattern for performance
            optimized_pattern = self.optimizer.optimize(pattern)
            
            # Compile and cache
            try:
                compiled = re.compile(optimized_pattern, re.IGNORECASE | re.MULTILINE)
                self.compiled_patterns[name] = compiled
            except re.error as e:
                print(f"Failed to compile pattern {name}: {e}")
    
    def scan_text(self, text: str, max_patterns: int = 10) -> list:
        """Scan text with optimized patterns."""
        if text in self.pattern_cache:
            return self.pattern_cache[text]
        
        results = []
        
        # Pre-filter patterns based on text characteristics
        relevant_patterns = self._filter_relevant_patterns(text)
        
        for name, pattern in list(relevant_patterns.items())[:max_patterns]:
            matches = pattern.findall(text)
            if matches:
                results.extend([{
                    "type": name,
                    "matches": matches,
                    "count": len(matches)
                }])
        
        # Cache results
        if len(self.pattern_cache) < 1000:  # Limit cache size
            self.pattern_cache[text] = results
        
        return results
    
    def _filter_relevant_patterns(self, text: str) -> dict:
        """Pre-filter patterns based on text content."""
        # Quick heuristics to avoid running unnecessary patterns
        relevant = {}
        
        # Check for digits (needed for SSN, credit cards, etc.)
        has_digits = any(c.isdigit() for c in text)
        
        # Check for @ symbol (needed for emails)
        has_at = '@' in text
        
        for name, pattern in self.compiled_patterns.items():
            if name in ['ssn', 'credit_card', 'phone'] and not has_digits:
                continue
            if name == 'email' and not has_at:
                continue
            
            relevant[name] = pattern
        
        return relevant

# Usage
matcher = OptimizedPatternMatcher()
matcher.add_patterns({
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
})

# Fast pattern matching
results = matcher.scan_text("Contact john@example.com for details")
```

## Caching Strategies

### Multi-Level Caching

**Level 1: Request Response Cache**
```python
import hashlib
import json
import time
from typing import Dict, Any, Optional
from threading import RLock

class RequestCache:
    """High-performance request cache with LRU eviction."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = RLock()
    
    def _generate_key(self, messages: list, model: str, **kwargs) -> str:
        """Generate cache key from request parameters."""
        # Include only cacheable parameters
        cache_params = {
            'messages': messages,
            'model': model,
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens')
        }
        
        key_str = json.dumps(cache_params, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(self, messages: list, model: str, **kwargs) -> Optional[Dict]:
        """Get cached response if available and not expired."""
        key = self._generate_key(messages, model, **kwargs)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if time.time() - entry['timestamp'] > self.ttl_seconds:
                    del self.cache[key]
                    del self.access_times[key]
                    return None
                
                # Update access time for LRU
                self.access_times[key] = time.time()
                return entry['response']
        
        return None
    
    def set(self, messages: list, model: str, response: Dict, **kwargs):
        """Cache response with LRU eviction."""
        key = self._generate_key(messages, model, **kwargs)
        
        with self.lock:
            # Evict oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                # Find least recently used key
                lru_key = min(self.access_times.keys(), 
                             key=lambda k: self.access_times[k])
                del self.cache[lru_key]
                del self.access_times[lru_key]
            
            # Add new entry
            self.cache[key] = {
                'response': response,
                'timestamp': time.time()
            }
            self.access_times[key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size * 100
            }

# Global cache instances
request_cache = RequestCache(max_size=5000, ttl_seconds=1800)  # 30 minutes
```

**Level 2: Compliance Detection Cache**
```python
class ComplianceCache:
    """Cache for compliance detection results."""
    
    def __init__(self, max_size: int = 50000, ttl_seconds: int = 7200):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.lock = RLock()
    
    def _text_hash(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:12]
    
    def get_detection_result(self, text: str) -> Optional[Dict]:
        """Get cached detection result."""
        key = self._text_hash(text)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if time.time() - entry['timestamp'] <= self.ttl_seconds:
                    return entry['result']
                else:
                    del self.cache[key]
        
        return None
    
    def cache_detection_result(self, text: str, result: Dict):
        """Cache detection result."""
        key = self._text_hash(text)
        
        with self.lock:
            # Simple eviction: remove oldest 10% when full
            if len(self.cache) >= self.max_size:
                sorted_keys = sorted(self.cache.keys(), 
                                   key=lambda k: self.cache[k]['timestamp'])
                for old_key in sorted_keys[:self.max_size // 10]:
                    del self.cache[old_key]
            
            self.cache[key] = {
                'result': result,
                'timestamp': time.time()
            }

compliance_cache = ComplianceCache(max_size=20000, ttl_seconds=3600)  # 1 hour
```

### Cache-Aware Client

```python
from deepsentinel import SentinelClient
import time

class CachedSentinelClient:
    """DeepSentinel client with integrated caching."""
    
    def __init__(self, **kwargs):
        self.client = SentinelClient(**kwargs)
        self.request_cache = RequestCache()
        self.compliance_cache = ComplianceCache()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def chat_completion_create(self, messages, **kwargs):
        """Create chat completion with caching."""
        model = kwargs.get('model', 'gpt-4o')
        
        # Check request cache first
        cached_response = self.request_cache.get(messages, model, **kwargs)
        if cached_response:
            self.cache_hits += 1
            cached_response['from_cache'] = True
            return cached_response
        
        self.cache_misses += 1
        
        # Make request
        start_time = time.time()
        response = self.client.chat.completions.create(messages=messages, **kwargs)
        duration = time.time() - start_time
        
        # Cache successful response
        result = {
            'response': response,
            'duration': duration,
            'from_cache': False
        }
        
        self.request_cache.set(messages, model, result, **kwargs)
        
        return result
    
    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests * 100 if total_requests > 0 else 0
        
        return {
            'request_cache': self.request_cache.get_stats(),
            'compliance_cache': {
                'size': len(self.compliance_cache.cache),
                'max_size': self.compliance_cache.max_size
            },
            'hit_rate': hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }
```

## Connection Optimization

### HTTP Connection Pooling

```python
import httpx
from deepsentinel import SentinelClient, SentinelConfig, NetworkConfig

# Configure optimized HTTP settings
network_config = NetworkConfig(
    connection_pool_size=100,        # Total connection pool size
    connection_pool_per_host=20,     # Connections per host
    connection_timeout=5.0,          # Connection establishment timeout
    read_timeout=30.0,               # Read timeout for requests
    write_timeout=5.0,               # Write timeout for uploads
    pool_timeout=1.0,                # Timeout to get connection from pool
    keep_alive_timeout=120,          # Keep alive timeout
    max_keepalive_connections=50,    # Max keep-alive connections
    enable_http2=True,               # Enable HTTP/2 for better multiplexing
    retries=3,                       # Automatic retries
    retry_backoff_factor=0.5         # Backoff factor for retries
)

config = SentinelConfig(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    network_config=network_config
)

client = SentinelClient(config=config)
```

### Async Client for High Concurrency

```python
import asyncio
from deepsentinel import AsyncSentinelClient
from typing import List, Dict

class HighThroughputClient:
    """High-throughput async client for DeepSentinel."""
    
    def __init__(self, **kwargs):
        self.client = AsyncSentinelClient(**kwargs)
        self.semaphore = asyncio.Semaphore(50)  # Limit concurrent requests
        
    async def process_batch(self, requests: List[Dict]) -> List[Dict]:
        """Process multiple requests concurrently."""
        async def process_single(request):
            async with self.semaphore:  # Limit concurrency
                try:
                    response = await self.client.chat.completions.create(**request)
                    return {
                        'success': True,
                        'response': response,
                        'request_id': request.get('id')
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e),
                        'request_id': request.get('id')
                    }
        
        # Process all requests concurrently
        tasks = [process_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def stream_process(self, requests: List[Dict], batch_size: int = 10):
        """Process requests in batches to avoid overwhelming the system."""
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            results = await self.process_batch(batch)
            
            for result in results:
                yield result
            
            # Brief pause between batches to be respectful
            await asyncio.sleep(0.1)

# Usage example
async def high_throughput_example():
    """Example of high-throughput processing."""
    client = HighThroughputClient(
        sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create batch of requests
    requests = [
        {
            'id': f'req_{i}',
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': f'Process item {i}'}],
            'max_tokens': 50
        }
        for i in range(100)
    ]
    
    # Process in batches
    results = []
    async for result in client.stream_process(requests, batch_size=20):
        results.append(result)
        
        if len(results) % 20 == 0:
            print(f"Processed {len(results)} requests...")
    
    # Analyze results
    successful = sum(1 for r in results if r['success'])
    print(f"Successfully processed {successful}/{len(results)} requests")

# Run with: asyncio.run(high_throughput_example())
```

## Request Batching and Optimization

### Intelligent Request Batching

```python
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Callable

@dataclass
class BatchRequest:
    """Individual request in a batch."""
    id: str
    messages: List[Dict]
    model: str
    callback: Callable
    kwargs: Dict
    timestamp: float

class RequestBatcher:
    """Intelligent request batcher for optimized throughput."""
    
    def __init__(self, 
                 batch_size: int = 10,
                 batch_timeout: float = 0.1,
                 max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_wait_time = max_wait_time
        
        self.pending_requests = []
        self.batches_by_model = defaultdict(list)
        self.last_batch_time = time.time()
        
    def add_request(self, request: BatchRequest):
        """Add request to batch queue."""
        self.pending_requests.append(request)
        self.batches_by_model[request.model].append(request)
        
        # Check if we should process a batch
        if self._should_process_batch():
            self._process_batches()
    
    def _should_process_batch(self) -> bool:
        """Determine if we should process pending batches."""
        current_time = time.time()
        
        # Process if any model has enough requests
        for model, requests in self.batches_by_model.items():
            if len(requests) >= self.batch_size:
                return True
        
        # Process if timeout exceeded
        if current_time - self.last_batch_time >= self.batch_timeout:
            return True
        
        # Process if max wait time exceeded for oldest request
        if self.pending_requests:
            oldest_request = min(self.pending_requests, key=lambda r: r.timestamp)
            if current_time - oldest_request.timestamp >= self.max_wait_time:
                return True
        
        return False
    
    def _process_batches(self):
        """Process pending batches."""
        for model, requests in self.batches_by_model.items():
            if requests:
                # Take up to batch_size requests
                batch = requests[:self.batch_size]
                
                # Remove from pending
                for req in batch:
                    self.pending_requests.remove(req)
                
                # Process batch asynchronously
                asyncio.create_task(self._process_batch(batch))
                
                # Update remaining requests
                self.batches_by_model[model] = requests[self.batch_size:]
        
        self.last_batch_time = time.time()
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests."""
        tasks = []
        
        for request in batch:
            task = asyncio.create_task(
                self._process_single_request(request)
            )
            tasks.append(task)
        
        # Wait for all requests in batch to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_request(self, request: BatchRequest):
        """Process a single request and call its callback."""
        try:
            # Make the actual API call here
            # This is a placeholder - you'd integrate with your actual client
            result = await self._make_api_call(request)
            request.callback(result)
        except Exception as e:
            request.callback({'error': str(e)})
    
    async def _make_api_call(self, request: BatchRequest):
        """Make the actual API call."""
        # Placeholder for actual API call
        await asyncio.sleep(0.1)  # Simulate API call
        return {'success': True, 'content': 'Response content'}
```

### Smart Request Prioritization

```python
from enum import Enum
from queue import PriorityQueue
import threading

class RequestPriority(Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0

@dataclass
class PrioritizedRequest:
    """Request with priority information."""
    priority: RequestPriority
    request: BatchRequest
    
    def __lt__(self, other):
        return self.priority.value < other.priority.value

class PriorityRequestProcessor:
    """Process requests based on priority."""
    
    def __init__(self, workers: int = 5):
        self.request_queue = PriorityQueue()
        self.workers = workers
        self.running = False
        self.worker_threads = []
    
    def start(self):
        """Start worker threads."""
        self.running = True
        for i in range(self.workers):
            thread = threading.Thread(target=self._worker, daemon=True)
            thread.start()
            self.worker_threads.append(thread)
    
    def stop(self):
        """Stop worker threads."""
        self.running = False
        # Add sentinel values to wake up workers
        for _ in range(self.workers):
            self.request_queue.put(None)
    
    def add_request(self, request: BatchRequest, priority: RequestPriority = RequestPriority.NORMAL):
        """Add request with priority."""
        prioritized_request = PrioritizedRequest(priority, request)
        self.request_queue.put(prioritized_request)
    
    def _worker(self):
        """Worker thread to process requests."""
        while self.running:
            try:
                item = self.request_queue.get(timeout=1)
                if item is None:  # Sentinel value
                    break
                
                # Process the request
                asyncio.run(self._process_request(item.request))
                
            except Exception as e:
                print(f"Error processing request: {e}")
    
    async def _process_request(self, request: BatchRequest):
        """Process individual request."""
        # Implementation would go here
        pass
```

## Performance Monitoring

### Comprehensive Performance Metrics

```python
import time
import statistics
from collections import deque, defaultdict
from threading import Lock
from typing import Dict, List, Any

class PerformanceMonitor:
    """Comprehensive performance monitoring for DeepSentinel."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.lock = Lock()
        
        # Metrics storage
        self.request_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.cache_hits = 0
        self.cache_misses = 0
        self.error_counts = defaultdict(int)
        self.model_stats = defaultdict(lambda: {
            'requests': 0,
            'total_time': 0,
            'tokens': 0
        })
        
        # Real-time metrics
        self.current_rps = 0  # Requests per second
        self.last_rps_calculation = time.time()
        self.recent_requests = deque(maxlen=100)
    
    def record_request(self, 
                      duration: float,
                      model: str,
                      tokens: int = 0,
                      cache_hit: bool = False,
                      detection_time: float = 0,
                      error: str = None):
        """Record request metrics."""
        with self.lock:
            current_time = time.time()
            
            # Record timing
            self.request_times.append(duration)
            if detection_time > 0:
                self.detection_times.append(detection_time)
            
            # Record cache metrics
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            
            # Record model-specific metrics
            self.model_stats[model]['requests'] += 1
            self.model_stats[model]['total_time'] += duration
            self.model_stats[model]['tokens'] += tokens
            
            # Record errors
            if error:
                self.error_counts[error] += 1
            
            # Update RPS calculation
            self.recent_requests.append(current_time)
            self._update_rps()
    
    def _update_rps(self):
        """Update requests per second calculation."""
        current_time = time.time()
        
        # Remove requests older than 1 second
        while (self.recent_requests and 
               current_time - self.recent_requests[0] > 1.0):
            self.recent_requests.popleft()
        
        self.current_rps = len(self.recent_requests)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self.lock:
            if not self.request_times:
                return {"message": "No data available"}
            
            # Calculate statistics
            request_stats = self._calculate_stats(list(self.request_times))
            detection_stats = self._calculate_stats(list(self.detection_times)) if self.detection_times else {}
            
            # Cache statistics
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
            
            # Model statistics
            model_summary = {}
            for model, stats in self.model_stats.items():
                avg_time = stats['total_time'] / stats['requests'] if stats['requests'] > 0 else 0
                model_summary[model] = {
                    'requests': stats['requests'],
                    'avg_response_time_ms': int(avg_time * 1000),
                    'total_tokens': stats['tokens'],
                    'avg_tokens_per_request': stats['tokens'] // stats['requests'] if stats['requests'] > 0 else 0
                }
            
            return {
                'request_performance': {
                    'total_requests': len(self.request_times),
                    'avg_response_time_ms': int(request_stats.get('mean', 0) * 1000),
                    'median_response_time_ms': int(request_stats.get('median', 0) * 1000),
                    'p95_response_time_ms': int(request_stats.get('p95', 0) * 1000),
                    'p99_response_time_ms': int(request_stats.get('p99', 0) * 1000),
                    'current_rps': self.current_rps
                },
                'detection_performance': {
                    'avg_detection_time_ms': int(detection_stats.get('mean', 0) * 1000),
                    'median_detection_time_ms': int(detection_stats.get('median', 0) * 1000)
                } if detection_stats else {'message': 'No detection data'},
                'cache_performance': {
                    'hit_rate_percent': round(cache_hit_rate, 2),
                    'hits': self.cache_hits,
                    'misses': self.cache_misses
                },
                'model_performance': model_summary,
                'error_summary': dict(self.error_counts)
            }
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics for a list of values."""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': sorted_values[int(n * 0.95)] if n > 0 else 0,
            'p99': sorted_values[int(n * 0.99)] if n > 0 else 0,
            'min': min(values),
            'max': max(values)
        }
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        with self.lock:
            recent_times = list(self.request_times)[-100:]  # Last 100 requests
            recent_stats = self._calculate_stats(recent_times) if recent_times else {}
            
            return {
                'current_rps': self.current_rps,
                'recent_avg_response_time_ms': int(recent_stats.get('mean', 0) * 1000),
                'active_requests': len(self.recent_requests),
                'cache_hit_rate_recent': self._calculate_recent_cache_hit_rate()
            }
    
    def _calculate_recent_cache_hit_rate(self) -> float:
        """Calculate cache hit rate for recent requests."""
        # This is a simplified calculation
        # In practice, you'd track cache metrics with timestamps
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0

# Global performance monitor
perf_monitor = PerformanceMonitor(window_size=5000)
```

### Performance Instrumentation

```python
import functools
import time
from typing import Callable, Any

def instrument_performance(monitor: PerformanceMonitor):
    """Decorator to instrument function performance."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = type(e).__name__
                raise
            finally:
                duration = time.time() - start_time
                
                # Extract relevant metrics from args/kwargs if available
                model = kwargs.get('model', 'unknown')
                tokens = getattr(result, 'usage.total_tokens', 0) if 'result' in locals() else 0
                
                monitor.record_request(
                    duration=duration,
                    model=model,
                    tokens=tokens,
                    error=error
                )
        
        return wrapper
    return decorator

# Usage
@instrument_performance(perf_monitor)
def monitored_completion(client, messages, **kwargs):
    """Completion function with performance monitoring."""
    return client.chat.completions.create(messages=messages, **kwargs)
```

## Memory Optimization

### Efficient Data Structures

```python
import sys
from typing import Dict, List, Any
import gc

class MemoryOptimizedCache:
    """Memory-efficient cache implementation."""
    
    def __init__(self, max_memory_mb: int = 100):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = {}
        self.key_sizes = {}
    
    def set(self, key: str, value: Any):
        """Set value with memory management."""
        # Calculate memory usage
        value_size = sys.getsizeof(value)
        key_size = sys.getsizeof(key)
        total_size = value_size + key_size
        
        # Check if we need to free memory
        if self._get_current_memory() + total_size > self.max_memory_bytes:
            self._evict_memory()
        
        self.cache[key] = value
        self.key_sizes[key] = total_size
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        return self.cache.get(key)
    
    def _get_current_memory(self) -> int:
        """Get current memory usage."""
        return sum(self.key_sizes.values())
    
    def _evict_memory(self):
        """Evict items to free memory."""
        # Simple LRU eviction - remove 25% of items
        items_to_remove = len(self.cache) // 4
        
        # In a real implementation, you'd track access times
        keys_to_remove = list(self.cache.keys())[:items_to_remove]
        
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
                del self.key_sizes[key]
        
        # Force garbage collection
        gc.collect()

class StreamingResponseHandler:
    """Memory-efficient streaming response handler."""
    
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size
        self.buffer = []
        self.buffer_size = 0
    
    def add_chunk(self, chunk: str) -> List[str]:
        """Add chunk and return completed segments."""
        self.buffer.append(chunk)
        self.buffer_size += len(chunk)
        
        completed_segments = []
        
        # If buffer exceeds chunk size, flush it
        if self.buffer_size >= self.chunk_size:
            segment = ''.join(self.buffer)
            completed_segments.append(segment)
            
            # Clear buffer
            self.buffer = []
            self.buffer_size = 0
        
        return completed_segments
    
    def flush(self) -> str:
        """Flush remaining buffer."""
        if self.buffer:
            segment = ''.join(self.buffer)
            self.buffer = []
            self.buffer_size = 0
            return segment
        return ""
```

## Scalability Patterns

### Horizontal Scaling with Load Balancing

```python
import random
from typing import List
import asyncio

class LoadBalancedClient:
    """Load-balanced DeepSentinel client for horizontal scaling."""
    
    def __init__(self, client_configs: List[Dict]):
        """Initialize with multiple client configurations."""
        self.clients = []
        for config in client_configs:
            client = SentinelClient(**config)
            self.clients.append({
                'client': client,
                'weight': config.get('weight', 1),
                'active_requests': 0,
                'total_requests': 0,
                'error_count': 0
            })
        
        self.current_index = 0
    
    def get_client(self, strategy: str = "round_robin"):
        """Get client based on load balancing strategy."""
        if strategy == "round_robin":
            return self._round_robin()
        elif strategy == "least_connections":
            return self._least_connections()
        elif strategy == "weighted_random":
            return self._weighted_random()
        else:
            return self._round_robin()
    
    def _round_robin(self):
        """Round-robin load balancing."""
        client_info = self.clients[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.clients)
        return client_info
    
    def _least_connections(self):
        """Least connections load balancing."""
        return min(self.clients, key=lambda c: c['active_requests'])
    
    def _weighted_random(self):
        """Weighted random load balancing."""
        total_weight = sum(c['weight'] for c in self.clients)
        r = random.uniform(0, total_weight)
        
        current_weight = 0
        for client_info in self.clients:
            current_weight += client_info['weight']
            if r <= current_weight:
                return client_info
        
        return self.clients[0]  # Fallback
    
    async def completion(self, messages: List[Dict], strategy: str = "round_robin", **kwargs):
        """Make completion request with load balancing."""
        client_info = self.get_client(strategy)
        client = client_info['client']
        
        # Track active requests
        client_info['active_requests'] += 1
        client_info['total_requests'] += 1
        
        try:
            response = await client.chat.completions.create(
                messages=messages, **kwargs
            )
            return response
        except Exception as e:
            client_info['error_count'] += 1
            raise
        finally:
            client_info['active_requests'] -= 1
    
    def get_client_stats(self) -> List[Dict]:
        """Get statistics for all clients."""
        stats = []
        for i, client_info in enumerate(self.clients):
            error_rate = (client_info['error_count'] / 
                         max(client_info['total_requests'], 1) * 100)
            
            stats.append({
                'client_id': i,
                'weight': client_info['weight'],
                'active_requests': client_info['active_requests'],
                'total_requests': client_info['total_requests'],
                'error_count': client_info['error_count'],
                'error_rate_percent': round(error_rate, 2)
            })
        
        return stats
```

## Performance Testing and Benchmarking

### Automated Performance Testing

```python
import asyncio
import time
import statistics
from typing import List, Dict

class PerformanceTester:
    """Automated performance testing for DeepSentinel."""
    
    def __init__(self, client):
        self.client = client
    
    async def run_load_test(self, 
                           concurrent_requests: int = 10,
                           total_requests: int = 100,
                           test_duration: int = 60) -> Dict:
        """Run comprehensive load test."""
        
        print(f"Starting load test: {concurrent_requests} concurrent, {total_requests} total")
        
        # Test scenarios
        scenarios = [
            {
                'name': 'simple_query',
                'messages': [{'role': 'user', 'content': 'Hello, how are you?'}],
                'weight': 0.5
            },
            {
                'name': 'complex_query',
                'messages': [{'role': 'user', 'content': 'Explain quantum computing in detail with examples and applications.'}],
                'weight': 0.3
            },
            {
                'name': 'compliance_test',
                'messages': [{'role': 'user', 'content': 'My email is test@example.com'}],
                'weight': 0.2
            }
        ]
        
        # Run test
        results = await self._execute_load_test(
            scenarios, concurrent_requests, total_requests, test_duration
        )
        
        return self._analyze_results(results)
    
    async def _execute_load_test(self, scenarios, concurrent_requests, total_requests, duration):
        """Execute the load test."""
        results = []
        semaphore = asyncio.Semaphore(concurrent_requests)
        start_time = time.time()
        
        async def make_request(scenario):
            async with semaphore:
                request_start = time.time()
                
                try:
                    response = await self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=scenario['messages'],
                        max_tokens=100
                    )
                    
                    request_end = time.time()
                    
                    return {
                        'scenario': scenario['name'],
                        'success': True,
                        'duration': request_end - request_start,
                        'tokens': response.usage.total_tokens,
                        'timestamp': request_start
                    }
                    
                except Exception as e:
                    request_end = time.time()
                    
                    return {
                        'scenario': scenario['name'],
                        'success': False,
                        'duration': request_end - request_start,
                        'error': str(e),
                        'timestamp': request_start
                    }
        
        # Generate requests based on weights
        tasks = []
        for _ in range(total_requests):
            scenario = self._select_weighted_scenario(scenarios)
            task = asyncio.create_task(make_request(scenario))
            tasks.append(task)
            
            # Stop if duration exceeded
            if time.time() - start_time > duration:
                break
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        return [r for r in results if isinstance(r, dict)]
    
    def _select_weighted_scenario(self, scenarios):
        """Select scenario based on weights."""
        total_weight = sum(s['weight'] for s in scenarios)
        r = random.uniform(0, total_weight)
        
        current_weight = 0
        for scenario in scenarios:
            current_weight += scenario['weight']
            if r <= current_weight:
                return scenario
        
        return scenarios[0]
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze test results."""
        if not results:
            return {'error': 'No results to analyze'}
        
        # Overall statistics
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        durations = [r['duration'] for r in successful_requests]
        tokens = [r['tokens'] for r in successful_requests if 'tokens' in r]
        
        # Calculate statistics
        duration_stats = self._calculate_stats(durations) if durations else {}
        
        # Per-scenario analysis
        scenario_stats = {}
        for scenario_name in set(r['scenario'] for r in results):
            scenario_results = [r for r in results if r['scenario'] == scenario_name]
            scenario_successful = [r for r in scenario_results if r['success']]
            scenario_durations = [r['duration'] for r in scenario_successful]
            
            scenario_stats[scenario_name] = {
                'total_requests': len(scenario_results),
                'successful_requests': len(scenario_successful),
                'success_rate': len(scenario_successful) / len(scenario_results) * 100,
                'avg_duration_ms': int(statistics.mean(scenario_durations) * 1000) if scenario_durations else 0,
                'median_duration_ms': int(statistics.median(scenario_durations) * 1000) if scenario_durations else 0
            }
        
        return {
            'summary': {
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate_percent': len(successful_requests) / len(results) * 100,
                'avg_response_time_ms': int(duration_stats.get('mean', 0) * 1000),
                'median_response_time_ms': int(duration_stats.get('median', 0) * 1000),
                'p95_response_time_ms': int(duration_stats.get('p95', 0) * 1000),
                'p99_response_time_ms': int(duration_stats.get('p99', 0) * 1000),
                'total_tokens': sum(tokens),
                'avg_tokens_per_request': sum(tokens) // len(tokens) if tokens else 0
            },
            'scenario_breakdown': scenario_stats,
            'error_analysis': self._analyze_errors(failed_requests)
        }
    
    def _calculate_stats(self, values: List[float]) -> Dict:
        """Calculate statistical metrics."""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': sorted_values[int(n * 0.95)] if n > 0 else 0,
            'p99': sorted_values[int(n * 0.99)] if n > 0 else 0,
            'min': min(values),
            'max': max(values)
        }
    
    def _analyze_errors(self, failed_requests: List[Dict]) -> Dict:
        """Analyze error patterns."""
        error_counts = {}
        for request in failed_requests:
            error = request.get('error', 'Unknown error')
            error_counts[error] = error_counts.get(error, 0) + 1
        
        return {
            'total_errors': len(failed_requests),
            'error_breakdown': error_counts,
            'most_common_error': max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
        }

# Usage
async def run_performance_test():
    """Run performance test suite."""
    client = SentinelClient(
        sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    tester = PerformanceTester(client)
    
    # Run load test
    results = await tester.run_load_test(
        concurrent_requests=20,
        total_requests=200,
        test_duration=120
    )
    
    print("Performance Test Results:")
    print(f"Success Rate: {results['summary']['success_rate_percent']:.1f}%")
    print(f"Average Response Time: {results['summary']['avg_response_time_ms']}ms")
    print(f"P95 Response Time: {results['summary']['p95_response_time_ms']}ms")
    print(f"Requests per Second: {results['summary']['total_requests'] / 120:.1f}")

# Run with: asyncio.run(run_performance_test())
```

## Performance Optimization Checklist

### Configuration Optimization ✅
- [ ] Connection pooling configured for high throughput
- [ ] Appropriate timeout values set
- [ ] Caching enabled and properly sized
- [ ] Compression enabled for large payloads

### Detection Optimization ✅
- [ ] Local-first detection for speed
- [ ] Pattern caching enabled
- [ ] Unnecessary patterns disabled
- [ ] Detection sensitivity tuned for use case

### Application Architecture ✅
- [ ] Async clients used for high concurrency
- [ ] Request batching implemented where applicable
- [ ] Load balancing across multiple instances
- [ ] Circuit breakers for resilience

### Monitoring and Alerting ✅
- [ ] Performance metrics collected
- [ ] Real-time monitoring dashboard
- [ ] Alerting for performance degradation
- [ ] Regular performance testing

### Resource Management ✅
- [ ] Memory usage optimized
- [ ] Connection pooling sized appropriately
- [ ] Garbage collection tuned
- [ ] Resource cleanup implemented

Following these optimization techniques should significantly improve your DeepSentinel application's performance while maintaining compliance coverage and reliability.

---

**Next Guide**: [Migration Guide →](migration.md)