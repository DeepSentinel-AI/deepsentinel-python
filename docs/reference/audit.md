# Audit System

The `deepsentinel.audit` module provides comprehensive audit logging capabilities for tracking compliance events, request/response data, and system activities.

## Overview

The audit system is responsible for:

- Logging compliance violations and policy decisions
- Recording LLM interactions for compliance purposes
- Tracking system events and activities
- Providing audit trails for security and debugging
- Supporting regulatory compliance requirements

## Components

The audit system consists of these primary components:

1. **Audit Client**: Main interface for logging audit events
2. **Event Definitions**: Structured event types for different audit events
3. **Storage Options**: Different storage backends for audit data

## Usage

The audit system is typically accessed through the `SentinelClient` instance:

```python
import asyncio
from deepsentinel import SentinelClient

async def main():
    client = SentinelClient(
        sentinel_api_key="your-sentinel-api-key",
        openai_api_key="your-openai-api-key",
        # Audit configuration
        audit_enabled=True,
        audit_log_level="INFO",
        audit_storage="file",  # "memory", "file", or "api"
        audit_file_path="./audit_logs/",
    )
    
    await client.initialize()
    
    # The audit system automatically logs events during normal operation
    response = await client.chat.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # Access audit client directly for custom events
    await client.audit_api.log_custom_event(
        event_type="USER_ACTION",
        message="User performed custom action",
        metadata={"action_id": "12345", "user_id": "user123"}
    )
    
    # Retrieve recent audit events
    recent_events = await client.audit_api.get_recent_events(limit=10)
    for event in recent_events:
        print(f"{event.timestamp} | {event.event_type}: {event.message}")
    
    await client.cleanup()

asyncio.run(main())
```

## AuditClient

The `AuditClient` class provides the main interface for audit operations.

```python
class AuditClient:
    def __init__(
        self,
        config: Union[Dict[str, Any], "AuditConfig"],
        api_client: Optional["DeepSentinelAPIClient"] = None
    ) -> None
```

**Parameters:**

- **config** (`Union[Dict[str, Any], AuditConfig]`): Audit configuration
- **api_client** (`DeepSentinelAPIClient`, optional): API client for cloud audit storage

### Methods

#### `log_request_event`

```python
async def log_request_event(
    self,
    request_id: str,
    provider: str,
    operation: str,
    request_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

Log a request to an LLM provider.

**Parameters:**

- **request_id** (`str`): Unique identifier for the request
- **provider** (`str`): Provider name
- **operation** (`str`): Operation type (e.g., "chat.create")
- **request_data** (`Dict[str, Any]`): Request data (sanitized of sensitive info)
- **metadata** (`Dict[str, Any]`, optional): Additional metadata

#### `log_response_event`

```python
async def log_response_event(
    self,
    request_id: str,
    provider: str,
    operation: str,
    response_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

Log a response from an LLM provider.

**Parameters:**

- **request_id** (`str`): Unique identifier for the request
- **provider** (`str`): Provider name
- **operation** (`str`): Operation type (e.g., "chat.create")
- **response_data** (`Dict[str, Any]`): Response data (sanitized of sensitive info)
- **metadata** (`Dict[str, Any]`, optional): Additional metadata

#### `log_compliance_event`

```python
async def log_compliance_event(
    self,
    policy_name: str,
    result: bool,
    severity: str,
    action: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

Log a compliance policy check result.

**Parameters:**

- **policy_name** (`str`): Name of the policy
- **result** (`bool`): Whether the check passed
- **severity** (`str`): Severity level
- **action** (`str`): Action taken
- **message** (`str`): Human-readable message
- **metadata** (`Dict[str, Any]`, optional): Additional metadata

#### `log_error_event`

```python
async def log_error_event(
    self,
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> None
```

Log an error event.

**Parameters:**

- **error_type** (`str`): Type of error
- **message** (`str`): Error message
- **details** (`Dict[str, Any]`, optional): Additional error details

#### `log_custom_event`

```python
async def log_custom_event(
    self,
    event_type: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

Log a custom event.

**Parameters:**

- **event_type** (`str`): Custom event type
- **message** (`str`): Event message
- **metadata** (`Dict[str, Any]`, optional): Additional metadata

#### `get_recent_events`

```python
async def get_recent_events(
    self,
    limit: int = 100,
    event_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[AuditEvent]
```

Retrieve recent audit events.

**Parameters:**

- **limit** (`int`, optional): Maximum number of events to return
- **event_type** (`str`, optional): Filter by event type
- **start_time** (`datetime`, optional): Start time filter
- **end_time** (`datetime`, optional): End time filter

**Returns:**

- `List[AuditEvent]`: List of audit events

#### `export_events`

```python
async def export_events(
    self,
    file_path: str,
    format: str = "json",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    event_types: Optional[List[str]] = None
) -> str
```

Export audit events to a file.

**Parameters:**

- **file_path** (`str`): Path to export file
- **format** (`str`, optional): Export format ("json", "csv")
- **start_time** (`datetime`, optional): Start time filter
- **end_time** (`datetime`, optional): End time filter
- **event_types** (`List[str]`, optional): Filter by event types

**Returns:**

- `str`: Path to the exported file

## Audit Events

The audit system uses structured event objects to represent different types of audit records.

### `AuditEvent`

Base class for all audit events.

```python
class AuditEvent(BaseModel):
    id: str
    timestamp: datetime
    event_type: str
    message: str
    metadata: Dict[str, Any] = {}
```

**Fields:**

- **id** (`str`): Unique identifier for the event
- **timestamp** (`datetime`): Time the event occurred
- **event_type** (`str`): Type of event
- **message** (`str`): Human-readable message
- **metadata** (`Dict[str, Any]`): Additional context information

### Event Types

The audit system defines several standard event types:

- **REQUEST**: LLM provider request
- **RESPONSE**: LLM provider response
- **COMPLIANCE_CHECK**: Compliance policy check
- **COMPLIANCE_VIOLATION**: Compliance policy violation
- **ERROR**: System error
- **AUTHENTICATION**: Authentication event
- **CONFIGURATION**: Configuration change
- **SYSTEM**: General system event
- **CUSTOM**: Custom user-defined event

## Configuration

The audit system can be configured through the `SentinelConfig` class:

```python
client = SentinelClient(
    # ... other configuration ...
    
    # Audit configuration
    audit_enabled=True,  # Enable/disable audit logging
    audit_log_level="INFO",  # Minimum log level for audit events
    audit_storage="file",  # Storage backend: "memory", "file", or "api"
    audit_file_path="./audit_logs/",  # Path for file storage
    audit_retention_days=90,  # Number of days to retain audit logs
    
    # Cloud audit configuration (when using "api" storage)
    sentinel_api_key="your-api-key",  # Required for cloud audit storage
)
```

### Storage Options

The audit system supports multiple storage backends:

#### Memory Storage

Stores audit events in memory. Useful for testing but events are lost when the application restarts.

```python
client = SentinelClient(
    audit_storage="memory",
    audit_memory_limit=1000  # Maximum number of events to keep in memory
)
```

#### File Storage

Stores audit events in local JSON files with rotation.

```python
client = SentinelClient(
    audit_storage="file",
    audit_file_path="./audit_logs/",
    audit_file_rotation="daily"  # "hourly", "daily", or "weekly"
)
```

#### API Storage (Cloud)

Sends audit events to the DeepSentinel cloud service for secure storage and management.

```python
client = SentinelClient(
    audit_storage="api",
    sentinel_api_key="your-api-key"  # Required for API storage
)
```

## Examples

### Basic Audit Configuration

```python
from deepsentinel import SentinelClient

# Configure audit with file storage
client = SentinelClient(
    sentinel_api_key="your-sentinel-api-key",
    openai_api_key="your-openai-api-key",
    audit_enabled=True,
    audit_storage="file",
    audit_file_path="./compliance_logs/"
)
```

### Retrieving Compliance Violations

```python
import asyncio
from deepsentinel import SentinelClient
from datetime import datetime, timedelta

async def get_compliance_violations():
    client = SentinelClient(
        sentinel_api_key="your-sentinel-api-key",
        openai_api_key="your-openai-api-key",
        audit_enabled=True
    )
    
    await client.initialize()
    
    # Get compliance violations from the last 24 hours
    yesterday = datetime.utcnow() - timedelta(days=1)
    
    events = await client.audit_api.get_recent_events(
        event_type="COMPLIANCE_VIOLATION",
        start_time=yesterday,
        limit=50
    )
    
    print(f"Found {len(events)} compliance violations in the last 24 hours:")
    for event in events:
        policy = event.metadata.get("policy_name", "Unknown")
        severity = event.metadata.get("severity", "Unknown")
        print(f"- {event.timestamp}: {policy} ({severity}): {event.message}")
    
    await client.cleanup()

asyncio.run(get_compliance_violations())
```

### Exporting Audit Logs

```python
import asyncio
from deepsentinel import SentinelClient
from datetime import datetime, timedelta

async def export_monthly_audit():
    client = SentinelClient(
        sentinel_api_key="your-sentinel-api-key",
        openai_api_key="your-openai-api-key",
        audit_enabled=True
    )
    
    await client.initialize()
    
    # Calculate the start and end of the previous month
    now = datetime.utcnow()
    first_day = datetime(now.year, now.month, 1)
    if now.month == 1:
        previous_month = datetime(now.year - 1, 12, 1)
    else:
        previous_month = datetime(now.year, now.month - 1, 1)
    
    # Export all events from the previous month
    export_path = await client.audit_api.export_events(
        file_path=f"./audit_exports/audit_{previous_month.strftime('%Y_%m')}.json",
        format="json",
        start_time=previous_month,
        end_time=first_day
    )
    
    print(f"Exported audit logs to: {export_path}")
    
    await client.cleanup()

asyncio.run(export_monthly_audit())
```

### Custom Event Logging

```python
import asyncio
import uuid
from deepsentinel import SentinelClient

async def track_user_actions():
    client = SentinelClient(
        sentinel_api_key="your-sentinel-api-key",
        openai_api_key="your-openai-api-key",
        audit_enabled=True
    )
    
    await client.initialize()
    
    # Log a custom user action event
    user_id = "user_12345"
    session_id = str(uuid.uuid4())
    
    await client.audit_api.log_custom_event(
        event_type="USER_LOGIN",
        message=f"User {user_id} logged in",
        metadata={
            "user_id": user_id,
            "session_id": session_id,
            "ip_address": "192.168.1.1",
            "login_method": "password"
        }
    )
    
    # Simulate user actions
    await client.audit_api.log_custom_event(
        event_type="USER_ACTION",
        message=f"User {user_id} accessed sensitive document",
        metadata={
            "user_id": user_id,
            "session_id": session_id,
            "document_id": "doc-9876",
            "document_type": "financial_report"
        }
    )
    
    await client.cleanup()

asyncio.run(track_user_actions())
```

## Best Practices

1. **Enable audit logging for compliance:** Always enable audit logging in production environments that require compliance tracking.

2. **Set appropriate retention periods:** Configure retention periods according to your regulatory requirements.

3. **Regular exports:** Schedule regular exports of audit logs for long-term storage or compliance reporting.

4. **Use cloud storage for scale:** For production systems with high volumes, use the cloud API storage backend.

5. **Add contextual metadata:** Include relevant context in metadata to make audit logs more useful.

6. **Monitor for violations:** Regularly review compliance violations to identify patterns and improve policies.

7. **Sanitize sensitive data:** Ensure sensitive data is properly redacted before logging to audit trails.
