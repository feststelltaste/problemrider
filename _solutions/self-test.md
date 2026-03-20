---
title: Self-Test
description: Ability of a component to check its own state and functionality
category:
- Testing
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/self-test
problems:
- monitoring-gaps
- unpredictable-system-behavior
- slow-incident-resolution
- inadequate-integration-tests
- deployment-risk
- system-outages
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Implement startup self-tests that verify critical dependencies, configurations, and data access before accepting traffic
- Add periodic self-tests that run during operation to detect drift or degradation in dependencies
- Include end-to-end smoke tests that exercise critical business paths with synthetic transactions
- Design self-tests to produce clear pass/fail results with diagnostic information on failure
- Integrate self-test results with health check endpoints and monitoring systems
- Use self-tests as deployment validation gates that prevent unhealthy instances from receiving traffic
- Keep self-tests fast and lightweight to avoid impacting system performance or startup time

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Catches configuration errors and dependency issues immediately at startup
- Provides continuous validation that the system can perform its core functions
- Reduces time spent diagnosing issues that self-tests can identify automatically
- Improves deployment confidence by validating each instance before it serves traffic

**Costs and Risks:**
- Self-tests that interact with external systems can cause side effects or load
- Slow self-tests delay startup and can interfere with rapid scaling
- Self-test maintenance adds ongoing effort as the system evolves
- False positive self-test failures can prevent healthy instances from starting

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy CRM system frequently failed after deployment due to missing environment variables, incorrect database connection strings, or unavailable third-party services. Engineers would spend 30 minutes diagnosing each failure manually. By adding startup self-tests that verified database connectivity, checked required environment variables, validated API keys against external services, and ran a synthetic customer lookup, the system detected configuration issues within seconds of starting and refused to accept traffic until all checks passed. Deployment failures that previously required manual investigation were now immediately diagnosed by the self-test output.
