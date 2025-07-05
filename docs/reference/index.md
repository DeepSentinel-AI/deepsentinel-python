# API Reference

Complete API documentation for the DeepSentinel Python SDK, automatically generated from source code docstrings.

## Core Components

### [SentinelClient](client.md)
The main client class for interacting with LLM providers through compliance middleware.

### [Configuration](config.md) 
Configuration classes for setting up compliance policies, provider settings, and SDK behavior.

### [Types & Models](types.md)
Data models and type definitions used throughout the SDK.

### [Exceptions](exceptions.md)
Exception hierarchy for handling various error conditions.

### [Interfaces](interfaces.md)
Interface definitions for different LLM operations (chat, completions, embeddings, etc.).

## Provider System

### [Provider Registry](providers/registry.md)
Central registry for managing multiple LLM provider connections.

### [OpenAI Provider](providers/openai.md)
OpenAI-specific provider adapter with full API compatibility.

### [Anthropic Provider](providers/anthropic.md) 
Anthropic Claude provider adapter for accessing Claude models.

### [Base Provider](providers.md)
Base provider interface and common provider functionality.

## Compliance System

### [Compliance Engine](compliance/engine.md)
Core compliance checking and policy enforcement system.

### [Compliance Policies](compliance/policies.md)
Policy management for defining compliance rules and actions.

### [Compliance Interceptor](compliance/interceptor.md)
Request/response interceptor for applying compliance checks.

### [Detection Engines](compliance/detection.md)
Specialized detection engines for PII, PHI, PCI, and custom patterns.

## Supporting Components

### [Audit System](audit.md)
Comprehensive audit logging and compliance tracking.

### [Caching](cache.md)
Performance optimization through intelligent caching of compliance decisions.

### [Metrics](metrics.md)
Performance and compliance metrics collection and reporting.

### [Middleware](middleware.md)
Core middleware infrastructure for request/response processing.

### [API Client](api.md)
HTTP client for communicating with DeepSentinel cloud services.

## Module Index

All SDK modules are documented with their public APIs, including classes, functions, and exceptions:

