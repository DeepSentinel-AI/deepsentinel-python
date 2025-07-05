# Compliance Setup Tutorial

This tutorial guides you through configuring and customizing DeepSentinel's compliance features to meet your organization's security requirements and regulatory obligations.

## Overview

By the end of this tutorial, you'll have:
- Created basic and advanced compliance policies
- Customized detection for different types of sensitive data
- Implemented industry-specific policies (HIPAA, PCI DSS, GDPR)
- Set up policy hierarchies and context-based policy selection
- Added audit logging and monitoring for compliance events

**Time required:** 30 minutes

## Prerequisites

- Python 3.8+ installed
- DeepSentinel SDK installed (`pip install deepsentinel-sdk`)
- OpenAI or Anthropic API key
- Basic understanding of [DeepSentinel basics](basic-integration.md)
- Understanding of relevant compliance requirements for your organization

## Step 1: Creating a Basic Compliance Policy

Let's start by creating a simple compliance policy:

```python
from deepsentinel import SentinelClient, CompliancePolicy
import os

# Create a basic policy with default settings
basic_policy = CompliancePolicy(
    name="basic-policy",
    description="Default policy with standard detection settings",
    
    # Enable standard detection types
    enable_pii_detection=True,    # Personal Identifiable Information
    enable_phi_detection=True,    # Protected Health Information
    enable_pci_detection=True,    # Payment Card Information
    
    # Default risk threshold (0-1, lower is stricter)
    max_risk_score=0.8,
    
    # Block requests when violations are found
    block_on_violation=True
)

# Initialize client with the policy
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    compliance_policies=[basic_policy]
)
```

This creates a standard policy that:
- Detects common types of sensitive data (PII, PHI, PCI)
- Blocks any request with a risk score above 0.8
- Applies to all interactions with the LLM provider

## Step 2: Testing Your Compliance Policy

Let's verify that the compliance policy is working correctly:

```python
# Test with a safe message
try:
    print("Testing with safe content...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": "What are the best practices for data security?"
        }]
    )
    print("✅ Request allowed")
    print(f"Response: {response.choices[0].message.content[:100]}...")
    
except ComplianceViolationError as e:
    print("❌ Request blocked:", e.message)

# Test with sensitive content
try:
    print("\nTesting with sensitive content...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": "My social security number is 123-45-6789 and my credit card is 4111-1111-1111-1111"
        }]
    )
    print("⚠️ Request was not blocked (unexpected)")
    
except ComplianceViolationError as e:
    print("✅ Request correctly blocked")
    print(f"Violation types: {[v.type for v in e.violations]}")
    print(f"Risk score: {e.risk_score}")
```

## Step 3: Configuring Detection Sensitivity

You can adjust how strict the detection is for different types of sensitive data:

```python
# Create a policy with customized detection levels
custom_sensitivity_policy = CompliancePolicy(
    name="custom-sensitivity-policy",
    description="Policy with customized detection sensitivity",
    
    # Enable detection types
    enable_pii_detection=True,
    enable_phi_detection=True,
    enable_pci_detection=True,
    
    # Set detection sensitivity for each type
    pii_detection_level="strict",     # Options: "relaxed", "moderate", "strict"
    phi_detection_level="moderate",   # More permissive for PHI
    pci_detection_level="strict",     # Very strict for payment data
    
    # Risk threshold
    max_risk_score=0.7,
    
    # Block violations
    block_on_violation=True
)
```

The detection levels control:
- **relaxed**: Only detects clear, high-confidence instances of sensitive data
- **moderate**: Balanced detection with reasonable confidence
- **strict**: Aggressive detection that may produce some false positives

## Step 4: Creating Industry-Specific Policies

Different industries have different compliance requirements. Let's create policies for specific industries:

### Healthcare Policy (HIPAA)

```python
# Healthcare policy for HIPAA compliance
healthcare_policy = CompliancePolicy(
    name="hipaa-policy",
    description="Strict healthcare policy for HIPAA compliance",
    
    # Enable healthcare-specific detection
    enable_pii_detection=True,
    enable_phi_detection=True,  # PHI detection is critical for healthcare
    enable_pci_detection=True,
    
    # Add HIPAA to jurisdictions
    jurisdictions=["HIPAA"],
    
    # Stricter risk threshold for healthcare
    max_risk_score=0.6,
    
    # Block on violation
    block_on_violation=True,
    
    # Healthcare-specific settings
    phi_detection_level="strict",
    custom_patterns=[
        r"(\b[A-Za-z]{2}\d{6}\b)",  # Medical record number pattern
        r"(\b\d{3}-\d{2}-\d{4}\b)"   # SSN pattern
    ],
    
    # Anonymize data in audit logs
    anonymize_audit_logs=True
)
```

### Financial Services Policy (PCI DSS)

```python
# Financial policy for PCI DSS compliance
financial_policy = CompliancePolicy(
    name="financial-policy",
    description="Financial services policy for PCI DSS compliance",
    
    # Enable financial-specific detection
    enable_pii_detection=True,
    enable_phi_detection=False,  # PHI less relevant for financial
    enable_pci_detection=True,   # PCI critical for financial services
    
    # Add relevant jurisdictions
    jurisdictions=["PCI-DSS", "GDPR"],
    
    # Financial risk threshold
    max_risk_score=0.7,
    
    # Block on violation
    block_on_violation=True,
    
    # Financial-specific settings
    pci_detection_level="strict",
    custom_patterns=[
        r"(\b(?:\d{4}[- ]){3}\d{4}\b)",  # Credit card pattern
        r"(\b\d{9,18}\b)"                # Account number pattern
    ],
    
    # Log all violations
    log_violations=True,
    log_violation_details=True
)
```

### General Business Policy (GDPR)

```python
# General business policy for GDPR compliance
general_policy = CompliancePolicy(
    name="gdpr-policy",
    description="General business policy for GDPR compliance",
    
    # Enable general detection
    enable_pii_detection=True,
    enable_phi_detection=False,
    enable_pci_detection=False,
    
    # Add GDPR jurisdiction
    jurisdictions=["GDPR"],
    
    # Standard risk threshold
    max_risk_score=0.8,
    
    # Anonymize instead of blocking
    block_on_violation=False,
    anonymize_sensitive_data=True,
    
    # PII-specific settings
    pii_detection_level="moderate",
    custom_patterns=[
        r"(\b[A-Z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)"  # Email
    ]
)
```

## Step 5: Data Anonymization Instead of Blocking

For some use cases, you might want to anonymize sensitive data rather than blocking the entire request:

```python
# Create a policy that anonymizes instead of blocks
anonymization_policy = CompliancePolicy(
    name="anonymization-policy",
    description="Policy that anonymizes sensitive data instead of blocking",
    
    # Enable standard detection types
    enable_pii_detection=True,
    enable_phi_detection=True,
    enable_pci_detection=True,
    
    # Standard risk threshold
    max_risk_score=0.8,
    
    # Key settings: Don't block, but anonymize instead
    block_on_violation=False,
    anonymize_sensitive_data=True,
    
    # Configure anonymization
    anonymization_char="*",         # Use asterisks for redaction
    preserve_formatting=True,       # Keep the format (like XXX-XX-XXXX for SSN)
    preserve_data_length=True       # Keep the same length
)

# Initialize client with the policy
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    compliance_policies=[anonymization_policy]
)

# Test with content containing sensitive data
try:
    print("\nSending message with sensitive data for anonymization...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": "My email is john.doe@example.com and my phone is 555-123-4567"
        }]
    )
    
    print("✅ Request processed with anonymization")
    print("Original had email and phone number")
    print(f"Anonymized: {response.request.messages[0].content}")
    print(f"Response: {response.choices[0].message.content[:100]}...")
    
except ComplianceViolationError as e:
    # This shouldn't happen since we're anonymizing, not blocking
    print("❌ Request was unexpectedly blocked:", e.message)
```

The anonymization settings determine:
- Which character is used for redaction (`anonymization_char`)
- Whether to preserve the format of sensitive data (`preserve_formatting`)
- Whether to maintain the same length as the original data (`preserve_data_length`)

## Step 6: Custom Pattern Detection

You can define custom patterns to detect organization-specific sensitive data:

```python
# Define custom patterns for organization-specific data
employee_id_pattern = r"\bEMP-\d{6}\b"  # Format: EMP-123456
project_code_pattern = r"\bPROJ-[A-Z]{2}-\d{4}\b"  # Format: PROJ-XY-1234
internal_code_pattern = r"\bINT-[A-Z0-9]{8}\b"  # Format: INT-AB12CD34

# Create policy with custom patterns
custom_pattern_policy = CompliancePolicy(
    name="custom-pattern-policy",
    description="Policy with custom patterns for organization-specific data",
    
    # Enable standard detection
    enable_pii_detection=True,
    
    # Standard risk threshold
    max_risk_score=0.8,
    
    # Add custom patterns with friendly names
    custom_patterns={
        "employee_id": employee_id_pattern,
        "project_code": project_code_pattern,
        "internal_code": internal_code_pattern
    },
    
    # Block violations
    block_on_violation=True
)

# Initialize client with the policy
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    compliance_policies=[custom_pattern_policy]
)

# Test with content containing custom pattern
try:
    print("\nTesting with content containing custom pattern...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": "Employee EMP-123456 is working on project PROJ-AB-1234"
        }]
    )
    print("⚠️ Custom pattern was not detected (unexpected)")
    
except ComplianceViolationError as e:
    print("✅ Custom pattern correctly detected")
    print(f"Violation types: {[v.type for v in e.violations]}")
```

## Step 7: Allowlists for Specific Data Types

In some cases, you might want to allow certain types of potentially sensitive data:

```python
# Create policy with allowlists
allowlist_policy = CompliancePolicy(
    name="allowlist-policy",
    description="Policy that allows certain types of potentially sensitive data",
    
    # Enable detection but with allowlists
    enable_pii_detection=True,
    enable_phi_detection=False,
    enable_pci_detection=True,
    
    # Define allowlists (data types that should be allowed)
    custom_allowlist=[
        "email_addresses",  # Allow email addresses
        "urls",             # Allow URLs
        "ip_addresses",     # Allow IP addresses
    ],
    
    # Block non-allowlisted sensitive data
    block_on_violation=True,
    max_risk_score=0.8
)

# Initialize client with the policy
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    compliance_policies=[allowlist_policy]
)

# Test with allowlisted data (email)
try:
    print("\nTesting with allowlisted data type (email)...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": "My email is john.doe@example.com and website is https://example.com"
        }]
    )
    print("✅ Request with allowlisted data correctly allowed")
    
except ComplianceViolationError as e:
    print("❌ Request was incorrectly blocked:", e.message)
```

## Step 8: Multiple Policies with Different Priorities

For complex organizations, you can set up multiple policies with different priorities:

```python
from deepsentinel import SentinelClient, SentinelConfig, CompliancePolicy

# Policy 1: Base policy (lowest priority)
base_policy = CompliancePolicy(
    name="base-policy",
    description="Default policy for all requests",
    enable_pii_detection=True,
    max_risk_score=0.9,  # Lenient
    priority=1,  # Lowest priority
    block_on_violation=False,
    anonymize_sensitive_data=True
)

# Policy 2: Finance department policy (medium priority)
finance_policy = CompliancePolicy(
    name="finance-policy",
    description="Policy for finance department",
    enable_pii_detection=True,
    enable_pci_detection=True,
    max_risk_score=0.7,  # Stricter
    priority=5,  # Medium priority
    block_on_violation=True,
    department="finance"  # Apply only to finance department
)

# Policy 3: Healthcare data policy (highest priority)
healthcare_policy = CompliancePolicy(
    name="healthcare-policy",
    description="Policy for healthcare data",
    enable_pii_detection=True,
    enable_phi_detection=True,
    max_risk_score=0.5,  # Very strict
    priority=10,  # Highest priority
    block_on_violation=True,
    data_categories=["healthcare"]  # Apply only to healthcare data
)

# Initialize client with all policies
config = SentinelConfig(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    compliance_policies=[base_policy, finance_policy, healthcare_policy]
)
client = SentinelClient(config=config)
```

Now you can specify which policy should apply based on metadata:

```python
# Example 1: General query (should use base policy)
try:
    metadata = {"data_type": "general"}
    print("\nGeneral query (should use base policy)...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello world"}],
        metadata=metadata  # Metadata determines which policy applies
    )
    print("✅ Using base policy")
except Exception as e:
    print(f"❌ Error: {str(e)}")

# Example 2: Finance query (should use finance policy)
try:
    metadata = {"department": "finance"}
    print("\nFinance department query (should use finance policy)...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Quarterly financial report"}],
        metadata=metadata
    )
    print("✅ Using finance policy")
except Exception as e:
    print(f"❌ Error: {str(e)}")
```

The policy selection is based on:
1. **Priority** - Higher priority policies are evaluated first
2. **Metadata** - Policies can be tagged with departments, data categories, or other attributes
3. **Specificity** - More specific policies take precedence over general ones

## Step 9: Add Audit Logging for Compliance Events

Comprehensive audit logging is crucial for compliance:

```python
import logging
from deepsentinel import SentinelClient, CompliancePolicy, AuditConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='compliance_audit.log'  # Log to file
)

# Create audit configuration
audit_config = AuditConfig(
    enabled=True,
    log_level="info",
    include_request_content=True,  # Log full request content
    include_response_content=True,  # Log full response content
    include_policy_decisions=True,  # Log policy decisions
    log_all_requests=True,          # Log all requests, not just violations
    destination="file",             # "file", "sentinel", or "both"
    file_path="compliance_audit.log"
)

# Create compliance policy with audit settings
policy = CompliancePolicy(
    name="audited-policy",
    enable_pii_detection=True,
    enable_phi_detection=True,
    enable_pci_detection=True,
    max_risk_score=0.8,
    block_on_violation=True,
    
    # Audit settings
    log_violations=True,
    log_violation_details=True,
    anonymize_audit_logs=True  # Redact sensitive data in logs
)

# Initialize client with policy and audit config
client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    compliance_policies=[policy],
    audit_config=audit_config
)
```

## Step 10: Complete Implementation Example

Let's put everything together in a comprehensive compliance setup:

```python
import os
import logging
from typing import List, Dict, Any
from deepsentinel import (
    SentinelClient,
    SentinelConfig,
    CompliancePolicy,
    AuditConfig,
    ComplianceViolationError
)

class ComplianceManager:
    """Helper class to manage compliance policies and auditing."""
    
    def __init__(self, api_keys: Dict[str, str], log_file: str = "compliance.log"):
        """Initialize the compliance manager.
        
        Args:
            api_keys: Dictionary with sentinel_api_key and provider keys
            log_file: Path to log file
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file
        )
        self.logger = logging.getLogger("compliance")
        
        # Create audit config
        audit_config = AuditConfig(
            enabled=True,
            log_level="info",
            include_request_content=True,
            include_policy_decisions=True,
            log_all_requests=True,
            destination="both",
            file_path=log_file
        )
        
        # Create policies
        policies = self._create_policies()
        
        # Create client config
        config = SentinelConfig(
            sentinel_api_key=api_keys["sentinel_api_key"],
            openai_api_key=api_keys.get("openai_api_key"),
            anthropic_api_key=api_keys.get("anthropic_api_key"),
            compliance_policies=policies,
            audit_config=audit_config
        )
        
        # Initialize client
        self.client = SentinelClient(config=config)
        self.logger.info("ComplianceManager initialized with %d policies", len(policies))
    
    def _create_policies(self) -> List[CompliancePolicy]:
        """Create compliance policies."""
        # Base policy (applies to all)
        base_policy = CompliancePolicy(
            name="base-policy",
            description="Default policy for all requests",
            enable_pii_detection=True,
            max_risk_score=0.9,
            priority=1,
            block_on_violation=False,
            anonymize_sensitive_data=True
        )
        
        # HIPAA policy
        hipaa_policy = CompliancePolicy(
            name="hipaa-policy",
            description="HIPAA compliance policy",
            enable_pii_detection=True,
            enable_phi_detection=True,
            max_risk_score=0.6,
            priority=10,
            block_on_violation=True,
            jurisdictions=["HIPAA"],
            phi_detection_level="strict",
            anonymize_audit_logs=True,
            data_categories=["healthcare", "medical"]
        )
        
        # PCI DSS policy
        pci_policy = CompliancePolicy(
            name="pci-policy",
            description="PCI DSS compliance policy",
            enable_pii_detection=True,
            enable_pci_detection=True,
            max_risk_score=0.7,
            priority=8,
            block_on_violation=True,
            jurisdictions=["PCI-DSS"],
            pci_detection_level="strict",
            data_categories=["financial", "payment"]
        )
        
        # GDPR policy
        gdpr_policy = CompliancePolicy(
            name="gdpr-policy",
            description="GDPR compliance policy",
            enable_pii_detection=True,
            max_risk_score=0.8,
            priority=5,
            block_on_violation=False,
            anonymize_sensitive_data=True,
            jurisdictions=["GDPR"],
            pii_detection_level="moderate",
            regions=["EU"]
        )
        
        return [base_policy, hipaa_policy, pci_policy, gdpr_policy]
    
    def process_request(self, messages: List[Dict], metadata: Dict = None, **kwargs) -> Dict:
        """Process a request with appropriate compliance policy.
        
        Args:
            messages: List of message objects
            metadata: Optional metadata for policy selection
            **kwargs: Additional parameters for the API call
            
        Returns:
            Dict containing response or error information
        """
        metadata = metadata or {}
        self.logger.info("Processing request with metadata: %s", metadata)
        
        try:
            # Make request with metadata for policy selection
            response = self.client.chat.completions.create(
                model=kwargs.get("model", "gpt-4o"),
                messages=messages,
                metadata=metadata,
                **kwargs
            )
            
            # Log success
            self.logger.info(
                "Request processed successfully. Model: %s, Tokens: %d",
                response.model,
                response.usage.total_tokens
            )
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model": response.model,
                "tokens": response.usage.total_tokens
            }
            
        except ComplianceViolationError as e:
            # Log compliance violation
            self.logger.warning(
                "Compliance violation: %s, Risk score: %.2f",
                e.message,
                e.risk_score
            )
            self.logger.warning(
                "Violations: %s",
                [v.type for v in e.violations]
            )
            
            return {
                "success": False,
                "error_type": "compliance_violation",
                "message": e.message,
                "violations": [v.type for v in e.violations],
                "risk_score": e.risk_score
            }
            
        except Exception as e:
            # Log unexpected errors
            self.logger.error("Unexpected error: %s", str(e))
            
            return {
                "success": False,
                "error_type": "unexpected_error",
                "message": str(e)
            }

# Usage example
def main():
    """Demo of the compliance manager."""
    # Get API keys from environment
    api_keys = {
        "sentinel_api_key": os.getenv("DEEPSENTINEL_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }
    
    # Initialize compliance manager
    manager = ComplianceManager(api_keys)
    
    # Test scenarios
    scenarios = [
        {
            "name": "General request",
            "messages": [{"role": "user", "content": "What are the best practices for data security?"}],
            "metadata": {"data_type": "general"},
        },
        {
            "name": "Healthcare request",
            "messages": [{"role": "user", "content": "Summarize treatment options for diabetes"}],
            "metadata": {"data_categories": ["healthcare"]},
        },
        {
            "name": "Financial request",
            "messages": [{"role": "user", "content": "Explain how credit card processing works"}],
            "metadata": {"data_categories": ["financial"]},
        },
        {
            "name": "EU user request",
            "messages": [{"role": "user", "content": "How does GDPR affect data storage?"}],
            "metadata": {"regions": ["EU"]},
        }
    ]
    
    # Process each scenario
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        result = manager.process_request(
            scenario["messages"],
            scenario["metadata"]
        )
        
        if result["success"]:
            print(f"✅ Success! Using model: {result['model']}")
            print(f"Response: {result['content'][:100]}...")
        else:
            print(f"❌ Error: {result['message']}")

if __name__ == "__main__":
    main()
```

## Compliance Integration with Existing Systems

To integrate DeepSentinel's compliance features with your existing systems:

### Logging to SIEM Systems

```python
from deepsentinel import SentinelClient, AuditConfig

# Configure integration with your SIEM system
audit_config = AuditConfig(
    enabled=True,
    destination="webhook",  # Send to external system
    webhook_url="https://your-siem-system.com/api/logs",
    webhook_headers={
        "Authorization": "Bearer your-token",
        "Content-Type": "application/json"
    },
    log_format="json",
    batch_size=10,  # Send logs in batches
    include_policy_decisions=True
)

client = SentinelClient(
    sentinel_api_key=os.getenv("DEEPSENTINEL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    audit_config=audit_config
)
```

### DLP Integration

```python
# Example of integrating with existing DLP systems
dlp_patterns = fetch_patterns_from_dlp_system()  # Your function to get patterns

# Create policy with patterns from DLP
dlp_policy = CompliancePolicy(
    name="dlp-integration-policy",
    enable_pii_detection=True,
    custom_patterns=dlp_patterns,
    max_risk_score=0.8,
    block_on_violation=True
)
```

## What's Next?

Congratulations! You've set up comprehensive compliance policies with DeepSentinel. Here are some next steps:

### Advanced Compliance Features
1. **Custom Detection Engines** - Create specialized detectors for your industry
2. **Real-time Policy Updates** - Implement dynamic policy updates
3. **Compliance Reporting** - Set up periodic compliance reports

### Next Topics to Explore
- **[Error Handling](error-handling.md)** - Gracefully handle compliance violations
- **[Best Practices](../guides/best-practices.md)** - Compliance best practices
- **[Performance Optimization](../guides/performance.md)** - Optimize compliance checking

## Common Issues

### False Positives
```
ComplianceViolationError: Detected SSN in content, but it was actually a product code
```
**Solution**: Add the pattern to an allowlist or reduce detection sensitivity

### Policy Selection Issues
```
Warning: No policy matched metadata {"department": "legal"}, using default policy
```
**Solution**: Ensure you've configured policies for all relevant departments/categories

### Performance Impact
```
High latency detected in compliance checking: 250ms average
```
**Solution**: See the [Performance Guide](../guides/performance.md) for optimization tips

## Getting Help

- **[Compliance Concepts](../concepts/compliance.md)** - Learn more about compliance concepts
- **[API Reference](../reference/compliance/)** - Complete compliance API documentation
- **[GitHub Issues](https://github.com/deepsentinel/deepsentinel-sdk/issues)** - Report bugs or request features

---

**Next Tutorial**: [Error Handling →](error-handling.md)