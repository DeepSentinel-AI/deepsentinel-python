# Tutorials

Step-by-step guides to help you integrate DeepSentinel into your applications. These tutorials cover common use cases and provide practical examples you can adapt for your needs.

## Getting Started Tutorials

### [Basic Integration](basic-integration.md)
Learn how to integrate DeepSentinel into an existing application that uses OpenAI's API. This tutorial covers the essential steps to add compliance checking with minimal code changes.

**What you'll learn:**
- How to replace OpenAI SDK calls with DeepSentinel
- Basic compliance configuration
- Error handling and logging
- Testing your integration

**Time:** 15 minutes

---

### [Multiple Providers](multiple-providers.md)
Set up DeepSentinel to work with multiple LLM providers (OpenAI, Anthropic, etc.) and learn how to route requests efficiently.

**What you'll learn:**
- Configuring multiple providers
- Provider selection strategies
- Load balancing and fallback handling
- Cost optimization techniques

**Time:** 20 minutes

---

### [Streaming Responses](streaming.md)
Implement streaming responses with compliance checking for real-time applications like chatbots and interactive assistants.

**What you'll learn:**
- Setting up streaming with DeepSentinel
- Handling compliance in streaming contexts
- Performance optimization for streams
- Error recovery in streaming scenarios

**Time:** 25 minutes

---

## Advanced Tutorials

### [Compliance Setup](compliance-setup.md)
Configure comprehensive compliance policies for your organization's specific requirements, including GDPR, HIPAA, and custom policies.

**What you'll learn:**
- Creating custom compliance policies
- Jurisdiction-specific configuration
- Risk score tuning
- Audit trail setup

**Time:** 30 minutes

---

### [Error Handling](error-handling.md)
Build robust applications that gracefully handle compliance violations, provider errors, and network issues.

**What you'll learn:**
- Comprehensive error handling strategies
- Compliance violation recovery
- Provider failover implementation
- Monitoring and alerting setup

**Time:** 25 minutes

---

## Tutorial Prerequisites

Before starting these tutorials, make sure you have:

- **Python 3.8+** installed
- **DeepSentinel API key** ([sign up here](https://deepsentinel.ai/signup))
- **LLM provider API keys** (OpenAI, Anthropic, etc.)
- **Basic Python knowledge** (variables, functions, error handling)

## Tutorial Structure

Each tutorial follows a consistent structure:

1. **Overview** - What you'll build and learn
2. **Prerequisites** - What you need before starting
3. **Step-by-step instructions** - Detailed implementation steps
4. **Code examples** - Complete, runnable code samples
5. **Testing** - How to verify your implementation works
6. **Next steps** - How to extend what you've learned

## Getting Help

If you get stuck during any tutorial:

- Check the [FAQ](../faq.md) for common issues
- Review the [API Reference](../reference/) for detailed documentation
- Open an issue on [GitHub](https://github.com/deepsentinel/deepsentinel-sdk/issues)
- Join our [Discord community](https://discord.gg/deepsentinel)

---

Ready to start? Begin with [Basic Integration →](basic-integration.md)