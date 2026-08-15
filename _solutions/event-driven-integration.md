---
title: Event-Driven Integration
description: Decoupling producers from consumers via asynchronous message broker communication
category:
- Architecture
problems:
- tight-coupling-issues
- high-coupling-low-cohesion
- monolithic-architecture-constraints
- integration-difficulties
- microservice-communication-overhead
- cross-system-data-synchronization-problems
- deployment-coupling
layout: solution
related_solutions:
- slug: event-driven-architecture
  similarity: 0.8
- slug: business-event-processing
  similarity: 0.8
- slug: asynchronous-processing
  similarity: 0.75
- slug: protocol-abstraction
  similarity: 0.75
- slug: adapter
  similarity: 0.7
- slug: api-gateway
  similarity: 0.7
---

## Description

Event-driven integration replaces direct, synchronous calls between systems with asynchronous messages published to and consumed from a broker, so that producers emit immutable facts about what has happened rather than issuing commands to consumers that must be available and responsive right now. This decouples the two sides both temporally and spatially: a consumer that is down, slow, or not yet built does not block the producer, and the broker buffers messages until the consumer catches up. In legacy systems built around long chains of synchronous calls between components, this coupling is often the direct cause of cascading failures, where one slow or unavailable downstream service degrades or breaks the entire request, and of the difficulty of adding new consumers without touching the producer's code. Introducing a broker such as Kafka or RabbitMQ at the highest-value integration points, and doing so incrementally rather than all at once, lets a team break this coupling gradually while keeping the legacy producer largely intact, though it also trades immediate consistency for eventual consistency and introduces new operational surface area — broker infrastructure, dead-letter queues, message ordering — that synchronous calls never had to deal with.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy order management system made synchronous HTTP calls to five downstream services during order processing. When any downstream service was slow or unavailable, orders failed. The team introduced Kafka as an event broker, with the order system publishing OrderPlaced events. Each downstream service consumed events independently and at its own pace. Order processing failures dropped from 5% to under 0.1%, and the team was later able to add a new analytics consumer without touching the order system at all.
