---
title: Resilience
description: Ability of a system to remain operational under adverse conditions or faults
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/reliability/resilience
problems:
- cascade-failures
- system-outages
- unpredictable-system-behavior
- brittle-codebase
- single-points-of-failure
- fear-of-change
- constant-firefighting
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Assess the current resilience posture by cataloging failure modes and their impact on the legacy system
- Implement circuit breakers, retries with exponential backoff, and timeouts at all integration points
- Design for partial availability so that failures in non-critical components do not affect core functionality
- Add bulkheads to isolate failures and prevent resource exhaustion from propagating across components
- Conduct chaos engineering experiments to discover and address unknown failure modes
- Build redundancy into critical paths and ensure failover mechanisms are tested regularly
- Create and maintain runbooks for known failure scenarios to enable rapid recovery

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- System continues serving users during partial failures instead of experiencing total outages
- Reduces business impact of infrastructure and software failures
- Builds team confidence to make changes knowing the system can tolerate failures
- Provides a systematic approach to improving legacy system reliability incrementally

**Costs and Risks:**
- Resilience patterns add complexity to already complex legacy codebases
- Testing resilience mechanisms requires dedicated effort and tooling
- Over-engineering resilience for non-critical components wastes resources
- Resilience mechanisms themselves can fail or cause unexpected behavior

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A large e-commerce platform experienced cascading failures every time its legacy inventory service became slow, because every other service waited indefinitely for inventory responses. By systematically adding circuit breakers, timeouts, and fallback behaviors at each integration point, the team transformed the system from one where any service failure caused total collapse to one where failures were contained and users experienced only minor feature degradation. The inventory service could now be deployed, restarted, or even fail completely without affecting the checkout flow, which used cached inventory data when the live service was unavailable.
