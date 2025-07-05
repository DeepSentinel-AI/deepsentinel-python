# Migration Guide

This guide provides step-by-step instructions for migrating existing applications from direct LLM provider SDKs to DeepSentinel, and upgrading between DeepSentinel versions with minimal disruption.

## Overview

This guide covers:
- Migration planning and strategy
- Gradual rollout techniques
- Compatibility considerations
- Testing migration success
- Rollback procedures
- Version-specific upgrade paths

**Time to read:** 30 minutes

## Migration Planning

### Pre-Migration Assessment

Before starting your migration, assess your current setup:

**Current LLM Usage Audit**
```python
# Create an audit script to understand your current usage
import re
import ast
import os
from typing import Dict, List, Set

class LLMUsageAuditor:
    """Audit existing codebase for LLM provider usage."""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.findings = {
            'openai_calls': [],
            'anthropic_calls': [],
            'api_key_locations': [],
            'model_references': set(),
            'streaming_usage': [],
            'function_calling': []
        }
    
    def audit_codebase(self) -> Dict:
        """Perform comprehensive audit of LLM usage."""
        print("🔍 Auditing codebase for LLM usage...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv']]
            
            for file in files:
                if file.endswith(('.py', '.js', '.ts')):
                    file_path = os.path.join(root, file)
                    self._audit_file(file_path)
        
        return self._generate_report()
    
    def _audit_file(self, file_path: str):
        """Audit individual file for LLM usage."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for OpenAI usage
            if 'import openai' in content or 'from openai' in content:
                self.findings['openai_calls'].append({
                    'file': file_path,
                    'import_type': 'openai'
                })
            
            # Check for Anthropic usage
            if 'import anthropic' in content or 'from anthropic' in content:
                self.findings['anthropic_calls'].append({
                    'file': file_path,
                    'import_type': 'anthropic'
                })
            
            # Find API key references
            api_key_patterns = [
                r'sk-[a-zA-Z0-9]{48}',  # OpenAI keys
                r'sk-ant-[a-zA-Z0-9-]{95}',  # Anthropic keys
                r'OPENAI_API_KEY',
                r'ANTHROPIC_API_KEY'
            ]
            
            for pattern in api_key_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    self.findings['api_key_locations'].append({
                        'file': file_path,
                        'pattern': pattern,
                        'matches': len(matches)
                    })
            
            # Find model references
            model_patterns = [
                r'gpt-[0-9]\.[0-9]',
                r'gpt-[0-9]o?',
                r'claude-[0-9]',
                r'text-davinci',
                r'text-embedding'
            ]
            
            for pattern in model_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                self.findings['model_references'].update(matches)
            
            # Check for streaming usage
            if 'stream=True' in content or 'stream: true' in content:
                self.findings['streaming_usage'].append(file_path)
            
            # Check for function calling
            if 'functions=' in content or 'tools=' in content:
                self.findings['function_calling'].append(file_path)
                
        except Exception as e:
            print(f"Error auditing {file_path}: {e}")
    
    def _generate_report(self) -> Dict:
        """Generate migration planning report."""
        total_files_with_llm = (
            len(self.findings['openai_calls']) + 
            len(self.findings['anthropic_calls'])
        )
        
        return {
            'summary': {
                'total_files_with_llm_usage': total_files_with_llm,
                'openai_files': len(self.findings['openai_calls']),
                'anthropic_files': len(self.findings['anthropic_calls']),
                'unique_models_used': len(self.findings['model_references']),
                'files_with_streaming': len(self.findings['streaming_usage']),
                'files_with_function_calling': len(self.findings['function_calling']),
                'api_key_references': len(self.findings['api_key_locations'])
            },
            'models_used': list(self.findings['model_references']),
            'migration_complexity': self._assess_complexity(),
            'recommendations': self._generate_recommendations(),
            'detailed_findings': self.findings
        }
    
    def _assess_complexity(self) -> str:
        """Assess migration complexity."""
        complexity_score = 0
        
        # Base complexity
        complexity_score += len(self.findings['openai_calls']) + len(self.findings['anthropic_calls'])
        
        # Add complexity for advanced features
        complexity_score += len(self.findings['streaming_usage']) * 2
        complexity_score += len(self.findings['function_calling']) * 3
        complexity_score += len(self.findings['model_references']) * 0.5
        
        if complexity_score < 5:
            return "LOW - Simple migration, few integration points"
        elif complexity_score < 15:
            return "MEDIUM - Moderate migration, some advanced features"
        else:
            return "HIGH - Complex migration, extensive LLM usage"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate migration recommendations."""
        recommendations = []
        
        if len(self.findings['openai_calls']) > 0:
            recommendations.append("Plan OpenAI to DeepSentinel migration for chat completions")
        
        if len(self.findings['anthropic_calls']) > 0:
            recommendations.append("Plan Anthropic to DeepSentinel migration")
        
        if len(self.findings['streaming_usage']) > 0:
            recommendations.append("Test streaming functionality thoroughly during migration")
        
        if len(self.findings['function_calling']) > 0:
            recommendations.append("Verify function calling compatibility with DeepSentinel")
        
        if len(self.findings['api_key_locations']) > 3:
            recommendations.append("Consolidate API key management using environment variables")
        
        return recommendations

# Usage
auditor = LLMUsageAuditor("/path/to/your/project")
report = auditor.audit_codebase()

print("\n📊 Migration Assessment Report")
print("=" * 50)
print(f"Files with LLM usage: {report['summary']['total_files_with_llm_usage']}")
print(f"Migration complexity: {report['migration_complexity']}")
print(f"Models in use: {', '.join(report['models_used'])}")
print("\n💡 Recommendations:")
for rec in report['recommendations']:
    print(f"  • {rec}")
```

### Migration Strategy Options

**Option 1: Big Bang Migration (Fastest)**
- Replace all LLM calls at once
- Best for small applications
- Higher risk but faster deployment

**Option 2: Gradual Migration (Recommended)**
- Migrate components incrementally
- Test each component before proceeding
- Lower risk, easier rollback

**Option 3: Parallel Running (Safest)**
- Run both systems simultaneously
- Gradually shift traffic to DeepSentinel
- Compare results before full cutover

## Gradual Migration Approach

### Phase 1: Foundation Setup

**Step 1: Install DeepSentinel**
```bash
# Install alongside existing dependencies
pip install deepsentinel-sdk

# Or add to requirements.txt
echo "deepsentinel-sdk>=1.0.0" >> requirements.txt
pip install -r requirements.txt
```

**Step 2: Configuration Management**
```python
# Create a migration-friendly configuration system
import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class MigrationConfig:
    """Configuration for gradual migration."""
    
    # Feature flags for migration
    use_deepsentinel: bool = False
    deepsentinel_percentage: float = 0.0  # 0-100% of traffic
    fallback_to_original: bool = True
    
    # API keys
    deepsentinel_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Migration settings
    enable_comparison_mode: bool = False  # Compare responses
    log_migration_events: bool = True
    
    @classmethod
    def from_environment(cls) -> 'MigrationConfig':
        """Load configuration from environment variables."""
        return cls(
            use_deepsentinel=os.getenv("USE_DEEPSENTINEL", "false").lower() == "true",
            deepsentinel_percentage=float(os.getenv("DEEPSENTINEL_PERCENTAGE", "0")),
            fallback_to_original=os.getenv("FALLBACK_TO_ORIGINAL", "true").lower() == "true",
            deepsentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            enable_comparison_mode=os.getenv("ENABLE_COMPARISON_MODE", "false").lower() == "true",
            log_migration_events=os.getenv("LOG_MIGRATION_EVENTS", "true").lower() == "true"
        )

# Global configuration
migration_config = MigrationConfig.from_environment()
```

**Step 3: Create Migration Wrapper**
```python
import random
import logging
import time
from typing import Dict, Any, List, Optional
from deepsentinel import SentinelClient
import openai  # Keep existing import

logger = logging.getLogger(__name__)

class MigrationClient:
    """Wrapper client for gradual migration to DeepSentinel."""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        
        # Initialize DeepSentinel client if enabled
        if config.use_deepsentinel and config.deepsentinel_api_key:
            self.deepsentinel_client = SentinelClient(
                sentinel_api_key=config.deepsentinel_api_key,
                openai_api_key=config.openai_api_key,
                anthropic_api_key=config.anthropic_api_key
            )
        else:
            self.deepsentinel_client = None
        
        # Keep original client
        if config.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
        else:
            self.openai_client = None
    
    def chat_completions_create(self, **kwargs) -> Dict[str, Any]:
        """Create chat completion with migration logic."""
        start_time = time.time()
        
        # Determine which client to use
        use_deepsentinel = self._should_use_deepsentinel()
        
        try:
            if use_deepsentinel and self.deepsentinel_client:
                logger.info("Using DeepSentinel for request")
                response = self._deepsentinel_request(**kwargs)
                
                # Optional: Compare with original in comparison mode
                if self.config.enable_comparison_mode:
                    self._compare_responses(kwargs, response)
                
                return self._format_response(response, "deepsentinel")
            
            else:
                logger.info("Using original OpenAI client for request")
                return self._original_request(**kwargs)
                
        except Exception as e:
            logger.error(f"Error with primary client: {e}")
            
            # Fallback logic
            if use_deepsentinel and self.config.fallback_to_original:
                logger.info("Falling back to original client")
                try:
                    return self._original_request(**kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise
            else:
                raise
    
    def _should_use_deepsentinel(self) -> bool:
        """Determine if DeepSentinel should be used for this request."""
        if not self.config.use_deepsentinel or not self.deepsentinel_client:
            return False
        
        # Use percentage-based rollout
        return random.random() * 100 < self.config.deepsentinel_percentage
    
    def _deepsentinel_request(self, **kwargs) -> Any:
        """Make request using DeepSentinel."""
        return self.deepsentinel_client.chat.completions.create(**kwargs)
    
    def _original_request(self, **kwargs) -> Dict[str, Any]:
        """Make request using original OpenAI client."""
        response = self.openai_client.chat.completions.create(**kwargs)
        return self._format_response(response, "openai")
    
    def _format_response(self, response: Any, client_type: str) -> Dict[str, Any]:
        """Format response with metadata."""
        if client_type == "deepsentinel":
            return {
                "response": response,
                "client_used": "deepsentinel",
                "compliance_checked": True,
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": response.usage
            }
        else:
            return {
                "response": response,
                "client_used": "openai",
                "compliance_checked": False,
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": response.usage
            }
    
    def _compare_responses(self, request_kwargs: Dict, deepsentinel_response: Any):
        """Compare DeepSentinel response with original (for testing)."""
        try:
            original_response = self.openai_client.chat.completions.create(**request_kwargs)
            
            # Log comparison for analysis
            logger.info("Response comparison", extra={
                "deepsentinel_content": deepsentinel_response.choices[0].message.content[:100],
                "original_content": original_response.choices[0].message.content[:100],
                "deepsentinel_tokens": deepsentinel_response.usage.total_tokens,
                "original_tokens": original_response.usage.total_tokens
            })
            
        except Exception as e:
            logger.error(f"Error in response comparison: {e}")

# Global migration client
migration_client = MigrationClient(migration_config)
```

### Phase 2: Incremental Migration

**Step 1: Start with Non-Critical Components**
```python
# Example: Migrate a simple chat function first
def chat_with_migration(user_message: str) -> str:
    """Chat function with migration support."""
    
    try:
        result = migration_client.chat_completions_create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=150
        )
        
        # Log migration metrics
        if migration_config.log_migration_events:
            logger.info("Migration event", extra={
                "client_used": result["client_used"],
                "compliance_checked": result["compliance_checked"],
                "success": True
            })
        
        return result["content"]
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise

# Usage remains the same
response = chat_with_migration("Hello, how are you?")
```

**Step 2: Gradual Traffic Increase**
```python
# Environment configuration for gradual rollout
# Week 1: 5% traffic
# export DEEPSENTINEL_PERCENTAGE=5

# Week 2: 15% traffic  
# export DEEPSENTINEL_PERCENTAGE=15

# Week 3: 35% traffic
# export DEEPSENTINEL_PERCENTAGE=35

# Week 4: 60% traffic
# export DEEPSENTINEL_PERCENTAGE=60

# Week 5: 85% traffic
# export DEEPSENTINEL_PERCENTAGE=85

# Week 6: 100% traffic
# export DEEPSENTINEL_PERCENTAGE=100
```

**Step 3: Monitor and Validate**
```python
import json
from datetime import datetime, timedelta
from collections import defaultdict

class MigrationMonitor:
    """Monitor migration progress and health."""
    
    def __init__(self):
        self.metrics = defaultdict(int)
        self.errors = []
        self.response_times = []
    
    def record_request(self, client_used: str, success: bool, 
                      response_time: float, error: str = None):
        """Record migration request metrics."""
        
        self.metrics[f"{client_used}_requests"] += 1
        
        if success:
            self.metrics[f"{client_used}_success"] += 1
            self.response_times.append({
                "client": client_used,
                "time": response_time,
                "timestamp": datetime.now()
            })
        else:
            self.metrics[f"{client_used}_errors"] += 1
            if error:
                self.errors.append({
                    "client": client_used,
                    "error": error,
                    "timestamp": datetime.now()
                })
    
    def get_migration_health(self) -> Dict[str, Any]:
        """Get migration health report."""
        
        total_deepsentinel = self.metrics["deepsentinel_requests"]
        total_openai = self.metrics["openai_requests"]
        total_requests = total_deepsentinel + total_openai
        
        if total_requests == 0:
            return {"status": "no_data"}
        
        # Calculate success rates
        deepsentinel_success_rate = (
            self.metrics["deepsentinel_success"] / max(total_deepsentinel, 1) * 100
        )
        openai_success_rate = (
            self.metrics["openai_success"] / max(total_openai, 1) * 100
        )
        
        # Calculate average response times
        recent_time = datetime.now() - timedelta(hours=1)
        recent_times = [r for r in self.response_times if r["timestamp"] > recent_time]
        
        deepsentinel_times = [r["time"] for r in recent_times if r["client"] == "deepsentinel"]
        openai_times = [r["time"] for r in recent_times if r["client"] == "openai"]
        
        return {
            "status": "healthy" if deepsentinel_success_rate > 95 else "warning",
            "traffic_split": {
                "deepsentinel_percentage": total_deepsentinel / total_requests * 100,
                "openai_percentage": total_openai / total_requests * 100
            },
            "success_rates": {
                "deepsentinel": deepsentinel_success_rate,
                "openai": openai_success_rate
            },
            "average_response_times": {
                "deepsentinel_ms": sum(deepsentinel_times) / len(deepsentinel_times) * 1000 if deepsentinel_times else 0,
                "openai_ms": sum(openai_times) / len(openai_times) * 1000 if openai_times else 0
            },
            "recent_errors": len([e for e in self.errors if e["timestamp"] > recent_time]),
            "recommendations": self._get_recommendations(deepsentinel_success_rate, openai_success_rate)
        }
    
    def _get_recommendations(self, ds_success: float, openai_success: float) -> List[str]:
        """Get recommendations based on metrics."""
        recommendations = []
        
        if ds_success < 90:
            recommendations.append("DeepSentinel success rate is low - investigate errors")
        
        if ds_success < openai_success - 5:
            recommendations.append("DeepSentinel performing worse than OpenAI - consider rollback")
        
        if ds_success > 95 and len(self.response_times) > 100:
            recommendations.append("Migration looking healthy - consider increasing traffic")
        
        return recommendations

# Global monitor
migration_monitor = MigrationMonitor()
```

### Phase 3: Advanced Feature Migration

**Streaming Migration**
```python
class StreamingMigrationClient:
    """Handle streaming migration specially."""
    
    def __init__(self, migration_client: MigrationClient):
        self.migration_client = migration_client
    
    def create_stream(self, **kwargs):
        """Create streaming completion with migration support."""
        kwargs['stream'] = True
        
        # For streaming, be more conservative about DeepSentinel usage
        if (self.migration_client.config.use_deepsentinel and 
            random.random() * 100 < self.migration_client.config.deepsentinel_percentage * 0.7):  # 70% of regular percentage
            
            try:
                return self.migration_client.deepsentinel_client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.warning(f"DeepSentinel streaming failed, falling back: {e}")
                if self.migration_client.config.fallback_to_original:
                    return self.migration_client.openai_client.chat.completions.create(**kwargs)
                raise
        else:
            return self.migration_client.openai_client.chat.completions.create(**kwargs)

# Usage
streaming_client = StreamingMigrationClient(migration_client)

def stream_chat_with_migration(messages: List[Dict]) -> str:
    """Streaming chat with migration support."""
    
    stream = streaming_client.create_stream(
        model="gpt-4o",
        messages=messages
    )
    
    response_content = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            response_content += content
    
    return response_content
```

**Function Calling Migration**
```python
def migrate_function_calling():
    """Example of migrating function calling."""
    
    # Define functions (same for both clients)
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }
    ]
    
    # Migration-aware function calling
    result = migration_client.chat_completions_create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather in New York?"}],
        tools=functions,
        tool_choice="auto"
    )
    
    # Handle response (same regardless of client used)
    if result["response"].choices[0].message.tool_calls:
        # Process function calls
        for tool_call in result["response"].choices[0].message.tool_calls:
            if tool_call.function.name == "get_weather":
                # Execute function
                pass
    
    return result["content"]
```

## Version-Specific Migration

### Upgrading from v0.x to v1.x

**Breaking Changes in v1.x**
```python
# OLD (v0.x) - Deprecated
from deepsentinel import DeepSentinel  # Old class name

client = DeepSentinel(api_key="...")  # Old constructor
response = client.complete(prompt="...")  # Old method name

# NEW (v1.x) - Current
from deepsentinel import SentinelClient  # New class name

client = SentinelClient(
    sentinel_api_key="...",  # Renamed parameter
    openai_api_key="..."     # Explicit provider keys
)
response = client.chat.completions.create(  # OpenAI-compatible interface
    model="gpt-4o",
    messages=[{"role": "user", "content": "..."}]
)
```

**Migration Script for v0.x to v1.x**
```python
import re
import os
from typing import List

class V0ToV1Migrator:
    """Automate migration from v0.x to v1.x."""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.changes_made = []
    
    def migrate_project(self):
        """Migrate entire project from v0.x to v1.x."""
        
        print("🔄 Starting migration from v0.x to v1.x...")
        
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self._migrate_file(file_path)
        
        print(f"✅ Migration completed. {len(self.changes_made)} files modified.")
        return self.changes_made
    
    def _migrate_file(self, file_path: str):
        """Migrate individual Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply migration transformations
            content = self._update_imports(content)
            content = self._update_class_usage(content)
            content = self._update_method_calls(content)
            content = self._update_constructor_calls(content)
            
            # Write back if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.changes_made.append(file_path)
                print(f"📝 Migrated: {file_path}")
                
        except Exception as e:
            print(f"❌ Error migrating {file_path}: {e}")
    
    def _update_imports(self, content: str) -> str:
        """Update import statements."""
        # Replace old import
        content = re.sub(
            r'from deepsentinel import DeepSentinel',
            'from deepsentinel import SentinelClient',
            content
        )
        
        return content
    
    def _update_class_usage(self, content: str) -> str:
        """Update class instantiation."""
        # Replace class name
        content = re.sub(
            r'\bDeepSentinel\(',
            'SentinelClient(',
            content
        )
        
        return content
    
    def _update_constructor_calls(self, content: str) -> str:
        """Update constructor parameter names."""
        # Update api_key parameter
        content = re.sub(
            r'SentinelClient\(\s*api_key\s*=',
            'SentinelClient(sentinel_api_key=',
            content
        )
        
        return content
    
    def _update_method_calls(self, content: str) -> str:
        """Update method calls."""
        # Replace .complete() with .chat.completions.create()
        content = re.sub(
            r'\.complete\(\s*prompt\s*=\s*([^,)]+)',
            r'.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": \1}]',
            content
        )
        
        return content

# Usage
migrator = V0ToV1Migrator("/path/to/your/project")
changes = migrator.migrate_project()
```

### Upgrading Between v1.x Versions

**v1.0 to v1.1 Migration**
```python
# New features in v1.1 - backwards compatible

# Enhanced compliance policies (new feature)
from deepsentinel import CompliancePolicy

# Old way (still works)
client = SentinelClient(
    sentinel_api_key="...",
    openai_api_key="..."
)

# New way (recommended)
policy = CompliancePolicy(
    name="my-policy",
    enable_pii_detection=True,
    max_risk_score=0.8
)

client = SentinelClient(
    sentinel_api_key="...",
    openai_api_key="...",
    compliance_policies=[policy]  # New parameter
)
```

**v1.1 to v1.2 Migration**
```python
# Performance improvements in v1.2

# New caching options (v1.2+)
from deepsentinel import SentinelClient, SentinelConfig, CacheConfig

cache_config = CacheConfig(
    enabled=True,
    ttl_seconds=300,
    max_size=1000
)

config = SentinelConfig(
    sentinel_api_key="...",
    openai_api_key="...",
    cache_config=cache_config  # New in v1.2
)

client = SentinelClient(config=config)
```

## Testing Migration Success

### Automated Migration Tests

```python
import pytest
import asyncio
from typing import Dict, Any

class MigrationTestSuite:
    """Comprehensive test suite for migration validation."""
    
    def __init__(self, original_client, migrated_client):
        self.original_client = original_client
        self.migrated_client = migrated_client
    
    async def run_compatibility_tests(self) -> Dict[str, Any]:
        """Run comprehensive compatibility tests."""
        
        results = {
            "basic_completion": await self._test_basic_completion(),
            "streaming": await self._test_streaming(),
            "function_calling": await self._test_function_calling(),
            "error_handling": await self._test_error_handling(),
            "performance": await self._test_performance(),
            "compliance": await self._test_compliance()
        }
        
        overall_success = all(r["success"] for r in results.values())
        
        return {
            "overall_success": overall_success,
            "individual_results": results,
            "summary": self._generate_test_summary(results)
        }
    
    async def _test_basic_completion(self) -> Dict[str, Any]:
        """Test basic completion compatibility."""
        test_message = "Hello, how are you?"
        
        try:
            # Test original
            original_response = self.original_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": test_message}],
                max_tokens=50
            )
            
            # Test migrated
            migrated_response = await self.migrated_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": test_message}],
                max_tokens=50
            )
            
            return {
                "success": True,
                "original_tokens": original_response.usage.total_tokens,
                "migrated_tokens": migrated_response.usage.total_tokens,
                "response_similarity": self._calculate_similarity(
                    original_response.choices[0].message.content,
                    migrated_response.choices[0].message.content
                )
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_streaming(self) -> Dict[str, Any]:
        """Test streaming compatibility."""
        try:
            # Test streaming with migrated client
            stream = await self.migrated_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Count to 5"}],
                stream=True,
                max_tokens=30
            )
            
            chunks_received = 0
            content_received = ""
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunks_received += 1
                    content_received += chunk.choices[0].delta.content
            
            return {
                "success": chunks_received > 0,
                "chunks_received": chunks_received,
                "content_length": len(content_received)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_function_calling(self) -> Dict[str, Any]:
        """Test function calling compatibility."""
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time"
                }
            }
        ]
        
        try:
            response = await self.migrated_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "What time is it?"}],
                tools=functions,
                tool_choice="auto"
            )
            
            has_function_call = (
                response.choices[0].message.tool_calls is not None and
                len(response.choices[0].message.tool_calls) > 0
            )
            
            return {
                "success": True,
                "function_called": has_function_call
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling compatibility."""
        try:
            # Test with invalid model
            await self.migrated_client.chat.completions.create(
                model="non-existent-model",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            return {
                "success": False,
                "error": "Expected error was not raised"
            }
            
        except Exception as e:
            # This is expected
            return {
                "success": True,
                "error_type": type(e).__name__
            }
    
    async def _test_performance(self) -> Dict[str, Any]:
        """Test performance comparison."""
        import time
        
        test_messages = [{"role": "user", "content": "Hello"}]
        
        # Time original client
        start_time = time.time()
        try:
            self.original_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=test_messages,
                max_tokens=10
            )
            original_time = time.time() - start_time
        except:
            original_time = None
        
        # Time migrated client
        start_time = time.time()
        try:
            await self.migrated_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=test_messages,
                max_tokens=10
            )
            migrated_time = time.time() - start_time
            
            return {
                "success": True,
                "original_time_ms": int(original_time * 1000) if original_time else None,
                "migrated_time_ms": int(migrated_time * 1000),
                "performance_ratio": migrated_time / original_time if original_time else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_compliance(self) -> Dict[str, Any]:
        """Test compliance functionality (new feature)."""
        try:
            # Test with sensitive data - should be blocked
            response = await self.migrated_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "My SSN is 123-45-6789"}]
            )
            
            return {
                "success": False,
                "error": "Compliance violation was not detected"
            }
            
        except Exception as e:
            # This is expected for compliance violation
            return {
                "success": True,
                "compliance_working": "ComplianceViolation" in str(type(e))
            }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified)."""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_test_summary(self, results: Dict[str, Dict]) -> str:
        """Generate human-readable test summary."""
        passed = sum(1 for r in results.values() if r["success"])
        total = len(results)
        
        summary = f"Migration tests: {passed}/{total} passed\n"
        
        for test_name, result in results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            summary += f"  {test_name}: {status}\n"
            
            if not result["success"] and "error" in result:
                summary += f"    Error: {result['error']}\n"
        
        return summary

# Usage
async def validate_migration():
    """Validate migration with comprehensive tests."""
    
    # Set up clients
    original_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    migrated_client = SentinelClient(
        sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Run tests
    test_suite = MigrationTestSuite(original_client, migrated_client)
    results = await test_suite.run_compatibility_tests()
    
    print("🧪 Migration Test Results")
    print("=" * 50)
    print(results["summary"])
    
    if results["overall_success"]:
        print("🎉 Migration validation successful!")
        return True
    else:
        print("⚠️ Migration validation found issues. Review before proceeding.")
        return False

# Run validation
success = asyncio.run(validate_migration())
```

## Rollback Procedures

### Automated Rollback System

```python
class MigrationRollback:
    """Handle rollback procedures for failed migrations."""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.rollback_triggers = {
            'error_rate_threshold': 10.0,  # % error rate
            'response_time_threshold': 5000,  # ms
            'success_rate_threshold': 90.0   # % success rate
        }
    
    def should_rollback(self, metrics: Dict[str, Any]) -> tuple[bool, str]:
        """Determine if rollback should be triggered."""
        
        # Check error rate
        error_rate = metrics.get('error_rate', 0)
        if error_rate > self.rollback_triggers['error_rate_threshold']:
            return True, f"Error rate too high: {error_rate}%"
        
        # Check response time
        avg_response_time = metrics.get('avg_response_time_ms', 0)
        if avg_response_time > self.rollback_triggers['response_time_threshold']:
            return True, f"Response time too high: {avg_response_time}ms"
        
        # Check success rate
        success_rate = metrics.get('success_rate', 100)
        if success_rate < self.rollback_triggers['success_rate_threshold']:
            return True, f"Success rate too low: {success_rate}%"
        
        return False, "All metrics within acceptable ranges"
    
    def execute_rollback(self, reason: str):
        """Execute rollback to previous configuration."""
        
        print(f"🚨 EXECUTING ROLLBACK: {reason}")
        
        # Step 1: Stop new DeepSentinel traffic
        self._set_environment_variable('DEEPSENTINEL_PERCENTAGE', '0')
        self._set_environment_variable('USE_DEEPSENTINEL', 'false')
        
        # Step 2: Enable fallback mode
        self._set_environment_variable('FALLBACK_TO_ORIGINAL', 'true')
        
        # Step 3: Log rollback event
        logger.critical("Migration rollback executed", extra={
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "rollback_config": {
                "deepsentinel_percentage": 0,
                "use_deepsentinel": False,
                "fallback_to_original": True
            }
        })
        
        print("✅ Rollback completed. All traffic routed to original client.")
    
    def _set_environment_variable(self, key: str, value: str):
        """Set environment variable (implementation depends on deployment)."""
        # This would typically update your deployment configuration
        # For example, updating Kubernetes ConfigMap, AWS Parameter Store, etc.
        
        print(f"Setting {key}={value}")
        os.environ[key] = value
        
        # In production, you might also need to:
        # - Update configuration service
        # - Restart application instances
        # - Update load balancer configuration

# Automated monitoring with rollback
class AutomatedMigrationMonitor:
    """Monitor migration and trigger rollback if needed."""
    
    def __init__(self, rollback_system: MigrationRollback):
        self.rollback_system = rollback_system
        self.monitoring_active = False
    
    def start_monitoring(self, check_interval: int = 60):
        """Start automated monitoring with rollback capability."""
        
        self.monitoring_active = True
        
        async def monitor_loop():
            while self.monitoring_active:
                try:
                    # Collect current metrics
                    metrics = self._collect_current_metrics()
                    
                    # Check if rollback is needed
                    should_rollback, reason = self.rollback_system.should_rollback(metrics)
                    
                    if should_rollback:
                        self.rollback_system.execute_rollback(reason)
                        self.monitoring_active = False
                        break
                    
                    # Log current status
                    logger.info("Migration monitoring check", extra=metrics)
                    
                except Exception as e:
                    logger.error(f"Error in migration monitoring: {e}")
                
                await asyncio.sleep(check_interval)
        
        # Start monitoring in background
        asyncio.create_task(monitor_loop())
        print(f"🔍 Migration monitoring started (check interval: {check_interval}s)")
    
    def stop_monitoring(self):
        """Stop automated monitoring."""
        self.monitoring_active = False
        print("⏹️ Migration monitoring stopped")
    
    def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        # This would integrate with your actual monitoring system
        # Example implementation:
        
        return {
            "error_rate": migration_monitor.metrics.get("error_rate", 0),
            "avg_response_time_ms": migration_monitor.metrics.get("avg_response_time", 0) * 1000,
            "success_rate": migration_monitor.metrics.get("success_rate", 100),
            "timestamp": datetime.now().isoformat()
        }

# Usage
rollback_system = MigrationRollback(migration_config)
automated_monitor = AutomatedMigrationMonitor(rollback_system)

# Start monitoring
automated_monitor.start_monitoring(check_interval=30)  # Check every 30 seconds
```

## Migration Checklist

### Pre-Migration ✅
- [ ] Audit current LLM usage
- [ ] Assess migration complexity
- [ ] Set up DeepSentinel account and API keys
- [ ] Plan migration strategy (gradual vs big bang)
- [ ] Set up monitoring and logging
- [ ] Create rollback procedures

### During Migration ✅
- [ ] Start with non-critical components
- [ ] Implement feature flags for traffic control
- [ ] Monitor error rates and performance
- [ ] Test each component thoroughly
- [ ] Document any issues encountered
- [ ] Gradually increase traffic percentage

### Post-Migration ✅
- [ ] Verify all functionality works correctly
- [ ] Check compliance features are active
- [ ] Monitor performance metrics
- [ ] Update documentation
- [ ] Train team on new features
- [ ] Remove old code after stable period

### Emergency Procedures ✅
- [ ] Rollback procedures tested and ready
- [ ] Monitoring alerts configured
- [ ] Contact information for DeepSentinel support
- [ ] Incident response plan documented

## Getting Help

If you encounter issues during migration:

- **[FAQ](../faq.md)** - Common migration questions
- **[GitHub Issues](https://github.com/deepsentinel/deepsentinel-sdk/issues)** - Report migration problems
- **[Discord Community](https://discord.gg/deepsentinel)** - Get help from other users
- **[Migration Support](mailto:migration@deepsentinel.ai)** - Direct migration assistance

---

**Congratulations!** You've completed your migration to DeepSentinel. Your application now has enterprise-grade compliance protection while maintaining the same familiar interface.