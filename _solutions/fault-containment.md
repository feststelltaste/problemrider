---
title: Fault Containment
description: Limiting the impact of faults to a small part of the system
category:
- Architecture
problems:
- cascade-failures
- single-points-of-failure
- ripple-effect-of-changes
- monolithic-architecture-constraints
- tight-coupling-issues
- unpredictable-system-behavior
- system-outages
layout: solution
related_solutions:
- slug: isolation-of-faulty-components
  similarity: 0.8
- slug: bulkhead
  similarity: 0.8
- slug: resilience
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.75
- slug: containerization
  similarity: 0.7
- slug: chaos-engineering
  similarity: 0.7
---

## Description

Fault containment limits the blast radius of a failure to the specific component where it originated, using techniques such as bulkhead isolation of resources, separate thread or connection pools per functional area, circuit breakers at integration boundaries, and process- or container-level isolation, so a fault in one part of the system cannot exhaust resources or propagate failures into unrelated parts. This addresses a defining characteristic of many legacy monoliths: components that were never designed with failure isolation in mind end up sharing a single process, memory space, and resource pool, so that a fault in a peripheral feature — a reporting module running out of memory, for instance — can take down an entirely unrelated critical path that happens to share the same server. Introducing isolation boundaries around the highest-risk components, with timeouts on every cross-component call so a slow dependency cannot silently block everything downstream of it, turns an all-or-nothing failure mode into a contained, recoverable one and creates natural seams that also support incremental modernization later. The cost is that retrofitting isolation into a monolith is itself substantial refactoring work, duplicates resources across the newly separate fault domains, and requires monitoring specifically built to detect faults that, by design, no longer surface as a full outage.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify fault domains in the legacy system by analyzing which components share resources, threads, or memory spaces
- Introduce bulkhead patterns to isolate critical subsystems so a failure in one does not consume resources needed by others
- Use separate thread pools, connection pools, or process boundaries for independent functional areas
- Apply circuit breakers at integration boundaries to stop fault propagation between services
- Deploy critical components in isolated containers or virtual machines to enforce process-level containment
- Add timeout policies to all cross-component calls to prevent a slow dependency from blocking the entire system
- Review error handling code to ensure exceptions are caught and handled locally rather than propagated unchecked

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Limits blast radius so a fault in one area does not take down the entire system
- Makes the system more predictable under partial failure conditions
- Enables independent recovery of failed components
- Supports incremental modernization by creating natural boundaries

**Costs and Risks:**
- Introducing isolation boundaries into a monolith requires significant refactoring effort
- Resource duplication across fault domains increases overall resource consumption
- Over-isolation can make legitimate cross-cutting operations more complex
- Teams need monitoring to detect and respond to contained faults that users may not notice

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A healthcare platform experienced full outages whenever its PDF report generation module ran out of memory, because it shared the same application server process as the patient records API. By moving report generation into a separate process with its own memory limits and a circuit breaker on the integration point, the team contained memory-related faults to the reporting subsystem. Patient record access remained available even during report generation failures, reducing the severity of incidents from critical to minor.
