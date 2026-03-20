---
title: Graceful Degradation
description: Ability of a system to operate in a limited capacity during failures or overload
category:
- Architecture
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/graceful-degradation
problems:
- system-outages
- cascade-failures
- unpredictable-system-behavior
- slow-application-performance
- capacity-mismatch
- constant-firefighting
- customer-dissatisfaction
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Classify system features by business criticality to determine which can be degraded and which must remain fully available
- Implement fallback behaviors for non-critical features (cached data, simplified responses, static content)
- Add load detection logic that activates degradation modes before the system reaches hard failure thresholds
- Design degradation to be transparent to users by displaying appropriate messaging about reduced functionality
- Test degradation modes regularly to ensure fallback paths actually work when needed
- Use feature toggles or configuration flags to manually trigger degradation during anticipated high-load events
- Monitor degradation state transitions and alert operations teams when the system enters reduced mode

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Core functionality remains available even during partial failures or overload
- Reduces the frequency and severity of complete system outages
- Provides a better user experience than hard failures or error pages
- Buys time for operations teams to address underlying issues

**Costs and Risks:**
- Designing and maintaining fallback paths adds development and testing effort
- Users may not realize they are receiving degraded functionality, leading to data inconsistencies
- Degradation logic can mask systemic problems that worsen over time
- Legacy systems may lack the architectural flexibility to support clean degradation boundaries

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An e-commerce platform built on a legacy monolith experienced complete outages during seasonal traffic spikes because its recommendation engine consumed excessive database resources. The team implemented graceful degradation by serving cached, non-personalized recommendations when database response times exceeded a threshold, and disabling recommendations entirely under extreme load. This kept the core shopping and checkout flow available during peak periods, converting what would have been full outages into minor feature reductions that most customers never noticed.
