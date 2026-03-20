---
title: Event-Driven Integration
description: Decoupling producers from consumers via asynchronous message broker communication
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/compatibility/event-driven-integration
problems:
- tight-coupling-issues
- high-coupling-low-cohesion
- monolithic-architecture-constraints
- integration-difficulties
- microservice-communication-overhead
- cross-system-data-synchronization-problems
- deployment-coupling
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify synchronous integration points between legacy systems that cause coupling or reliability issues
- Introduce a message broker (e.g., Kafka, RabbitMQ) and have producers emit domain events instead of making direct calls
- Design events as immutable facts about what happened, not commands for what should happen
- Add event publishing to legacy systems incrementally, starting with the highest-value or most painful integration points
- Implement idempotent consumers to handle message redelivery gracefully
- Use event schemas with a registry to maintain compatibility as events evolve

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Decouples systems temporally and spatially: producers and consumers do not need to be available simultaneously
- Enables adding new consumers without modifying the producer, supporting incremental modernization
- Improves resilience by buffering messages during consumer downtime

**Costs and Risks:**
- Introduces eventual consistency, which can be challenging for workflows that expect immediate data availability
- Adds operational complexity through broker infrastructure, monitoring, and dead-letter queue management
- Debugging asynchronous flows is harder than tracing synchronous request-response chains
- Message ordering and exactly-once delivery guarantees vary by broker and require careful design

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy order management system made synchronous HTTP calls to five downstream services during order processing. When any downstream service was slow or unavailable, orders failed. The team introduced Kafka as an event broker, with the order system publishing OrderPlaced events. Each downstream service consumed events independently and at its own pace. Order processing failures dropped from 5% to under 0.1%, and the team was later able to add a new analytics consumer without touching the order system at all.
