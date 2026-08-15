---
title: Microservices
description: Enabling rapid product experimentation through independent, business-aligned
  services
category:
- Architecture
problems:
- monolithic-architecture-constraints
- deployment-coupling
- tight-coupling-issues
- slow-feature-development
- scaling-inefficiencies
- increased-time-to-market
- large-risky-releases
- stagnant-architecture
- team-silos
layout: solution
related_solutions:
- slug: microservices-architecture
  similarity: 0.9
- slug: strangler-fig-pattern
  similarity: 0.75
- slug: modularization-and-bounded-contexts
  similarity: 0.75
- slug: event-driven-architecture
  similarity: 0.75
- slug: service-mesh
  similarity: 0.7
- slug: business-event-processing
  similarity: 0.7
---

## Description

Microservices are independently deployable, business-capability-aligned services that let separate teams design, build, release, and scale their part of a system on their own schedule, coordinating through explicit API contracts instead of through the shared codebase and shared release calendar. Adopting this approach in a legacy context is done through incremental extraction rather than a rewrite: boundaries are identified where data sharing and business lifecycle are naturally minimal, functionality is peeled off into a new service one step at a time using the strangler fig pattern, and observability is put in place before extraction begins because a distributed system without tracing and centralized logs is far harder to debug than the monolith it replaces. Legacy monoliths tend to force every team into the same deployment cadence and the same technology stack regardless of whether that fits the problem each team is actually solving, which slows time to market and prevents any one part of the system from being scaled, released, or modernized independently of the rest. Restructuring around microservices removes that shared bottleneck, letting a team rewrite or scale its own service without coordinating with every other team, but the degree of decomposition matters enormously in practice: extracting too many, too finely-grained services from a system whose data coupling was never well understood tends to replace a slow but comprehensible monolith with a fast-failing web of synchronous service calls that is considerably harder to reason about and operate. The realistic legacy modernization path therefore favors coarser, business-aligned services extracted gradually and validated one at a time, rather than a big-bang decomposition into as many services as the domain model can theoretically support.

## How to Apply ◆

> Decomposing a legacy monolith into microservices is one of the most common — and most frequently botched — modernization strategies. Success requires careful boundary identification and incremental extraction.

- Identify natural service boundaries by analyzing the legacy system's domain model, looking for areas with minimal data sharing and independent business lifecycles.
- Use the strangler fig pattern to extract services incrementally rather than rewriting the monolith from scratch — route specific functionality to new services while the monolith continues to handle everything else.
- Start with the least coupled, most well-understood part of the system to build team experience before tackling core business logic.
- Establish clear API contracts between services from the start, using contract testing to prevent integration failures as the number of services grows.
- Implement service-level observability (distributed tracing, centralized logging, health checks) before extracting the first service, because debugging distributed systems without observability is significantly harder than debugging a monolith.
- Resist the urge to create fine-grained services — in legacy contexts, larger services aligned to business capabilities are usually more manageable than dozens of tiny services.
- Plan for data ownership carefully: each service should own its data store, and shared databases must be eliminated through explicit data synchronization or event-driven approaches.

## Tradeoffs ⇄

> Microservices trade monolith complexity for distributed system complexity — the net benefit depends on whether the team has the infrastructure and skills to manage the latter.

**Benefits:**

- Enables independent deployment of services, allowing teams to release changes to one part of the system without coordinating with every other team.
- Allows different parts of the system to scale independently based on actual demand rather than scaling the entire monolith.
- Provides natural team boundaries aligned to business capabilities, reducing coordination overhead.
- Enables incremental technology modernization — individual services can be rewritten or upgraded without affecting the rest of the system.

**Costs and Risks:**

- Introduces distributed system complexity including network failures, eventual consistency, and debugging challenges that monoliths do not have.
- Requires significant infrastructure investment in service discovery, API gateways, container orchestration, and monitoring.
- Premature decomposition of a poorly understood legacy system often creates distributed monoliths that are harder to maintain than the original.
- Data consistency across service boundaries requires careful design and often introduces eventual consistency patterns that the team may not be experienced with.
- Operational overhead increases substantially — each service needs its own deployment pipeline, monitoring, and on-call rotation.

## How It Could Be

> The following scenarios illustrate both successful and cautionary microservices adoption in legacy contexts.

A logistics company with a 12-year-old monolithic shipment tracking application began its decomposition by extracting the notification subsystem into a standalone service. This was a natural first candidate because notifications had a clear interface (shipment events in, messages out) and minimal shared state with the rest of the system. The extraction took six weeks and gave the team experience with service deployment, inter-service communication, and distributed tracing. Over the following 18 months, the team extracted four more services, each time applying lessons learned from the previous extraction.

A retail company attempted to decompose its order management monolith into 30 microservices in a single six-month project. The team underestimated the data coupling between components and ended up with services that made synchronous calls to each other in long chains, creating cascading failure scenarios that were far worse than anything the monolith had experienced. After a series of production outages, the team consolidated back to eight coarser-grained services aligned to business domains, which proved far more manageable.
