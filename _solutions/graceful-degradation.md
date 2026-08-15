---
title: Graceful Degradation
description: Ability of a system to operate in a limited capacity during failures
  or overload
category:
- Architecture
- Operations
problems:
- system-outages
- cascade-failures
- unpredictable-system-behavior
- slow-application-performance
- capacity-mismatch
- constant-firefighting
- customer-dissatisfaction
- upstream-timeouts
layout: solution
related_solutions:
- slug: resilience
  similarity: 0.75
- slug: failover-mechanisms
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.75
- slug: load-shedding
  similarity: 0.75
- slug: redundancy
  similarity: 0.7
- slug: rollback-mechanisms
  similarity: 0.7
---

## Description

Graceful degradation is the design property that allows a system to continue serving its most important functions in a reduced capacity when parts of it fail or become overloaded, rather than failing completely. It works by classifying features according to business criticality, defining fallback behaviors for the less critical ones — cached data, simplified responses, disabled non-essential features — and detecting overload or partial failure early enough to switch into a degraded mode before the system reaches a hard failure threshold. This differs from redundancy or failover, which aim to keep full functionality available by masking a failure entirely; graceful degradation instead accepts a visible, controlled reduction in service as the deliberate alternative to an uncontrolled outage. Legacy systems are frequently vulnerable to complete outages precisely because components that should be independent — a recommendation engine and the checkout flow, for instance — share resources or fail-fast code paths that were never designed with isolation in mind, so a spike in load on a peripheral feature can take down the entire application. Introducing graceful degradation into such a system means retrofitting boundaries around non-essential functionality so that its failure or throttling cannot cascade into core workflows, converting what would otherwise be a full outage into a temporary, contained loss of secondary functionality that most users may not even notice.

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
