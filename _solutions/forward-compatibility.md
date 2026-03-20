---
title: Forward Compatibility
description: Ensure compatibility of existing systems with future versions
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/compatibility/forward-compatibility
problems:
- breaking-changes
- fear-of-change
- stagnant-architecture
- technical-architecture-limitations
- integration-difficulties
- technology-lock-in
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Design data formats and protocols to tolerate unknown fields and values by ignoring rather than rejecting them
- Use extensible schemas (e.g., optional fields, extension points) that can accommodate future additions
- Implement the robustness principle: be conservative in what you send, liberal in what you accept
- Design APIs with extension points such as custom headers or metadata fields for future use
- Test systems against hypothetical future versions by adding unknown fields and verifying they are handled gracefully
- Avoid tight coupling to specific enum values or status codes that may be extended in future versions

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces the frequency of forced upgrades when new versions are released
- Enables producers to evolve without waiting for all consumers to update
- Extends the useful life of deployed systems by accommodating change gracefully

**Costs and Risks:**
- Tolerant parsing can mask real errors by silently ignoring data that should cause failures
- Designing for unknown futures adds upfront complexity that may never be needed
- Forward-compatible systems may accumulate stale data or behaviors that confuse users
- Testing forward compatibility is inherently speculative and cannot cover all scenarios

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A payment gateway designed its transaction response format to include a set of known status codes but also instructed consumers to treat any unknown status code as "pending" rather than failing. When the gateway later added three new status codes for regulatory compliance, 90% of consumers handled them gracefully without any code changes. The remaining 10% that had implemented strict enum validation required emergency patches, reinforcing the value of the forward-compatible design.
