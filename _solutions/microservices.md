---
title: Microservices
description: Enabling rapid product experimentation through independent, business-aligned services
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/microservices
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
---

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

## Examples

> The following scenarios illustrate both successful and cautionary microservices adoption in legacy contexts.

A logistics company with a 12-year-old monolithic shipment tracking application began its decomposition by extracting the notification subsystem into a standalone service. This was a natural first candidate because notifications had a clear interface (shipment events in, messages out) and minimal shared state with the rest of the system. The extraction took six weeks and gave the team experience with service deployment, inter-service communication, and distributed tracing. Over the following 18 months, the team extracted four more services, each time applying lessons learned from the previous extraction.

A retail company attempted to decompose its order management monolith into 30 microservices in a single six-month project. The team underestimated the data coupling between components and ended up with services that made synchronous calls to each other in long chains, creating cascading failure scenarios that were far worse than anything the monolith had experienced. After a series of production outages, the team consolidated back to eight coarser-grained services aligned to business domains, which proved far more manageable.
