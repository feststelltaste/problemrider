---
title: Business Event Processing
description: Recognize, process, and respond to business events
category:
- Architecture
- Business
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/business-event-processing
problems:
- monolithic-architecture-constraints
- tight-coupling-issues
- legacy-business-logic-extraction-difficulty
- slow-application-performance
- cascade-failures
- complex-and-obscure-logic
layout: solution
---

## How to Apply ◆

- Identify key business events in the legacy system (order placed, payment received, shipment dispatched) and model them explicitly rather than embedding them in procedural flows.
- Introduce an event bus or message broker (Kafka, RabbitMQ) to decouple event producers from consumers in the legacy architecture.
- Refactor synchronous, tightly coupled legacy workflows into event-driven flows incrementally, starting with the most problematic integration points.
- Define event schemas and ensure they carry enough context for consumers to process them independently.
- Implement event sourcing for critical business processes where audit trail and state reconstruction are needed.
- Add monitoring and alerting for event processing to detect delays or failures.

## Tradeoffs ⇄

**Benefits:**
- Decouples business processes, allowing independent scaling and evolution of producers and consumers.
- Makes business logic more explicit and traceable through event streams.
- Enables real-time reactions to business events that legacy batch processing cannot support.
- Facilitates gradual decomposition of monolithic legacy systems.

**Costs:**
- Introduces eventual consistency, which legacy systems designed for immediate consistency may not handle well.
- Event-driven architectures are harder to debug and reason about than synchronous call chains.
- Requires infrastructure for reliable event delivery and processing.
- Retrofitting event processing into a legacy system requires careful identification of implicit events.

## How It Could Be

A legacy retail system processes orders through a monolithic transaction that spans inventory, billing, and shipping in a single database transaction. When any step fails, the entire order fails. The team introduces an event-driven approach: order placement emits an "OrderCreated" event, and inventory, billing, and shipping each subscribe independently. Each service handles its part asynchronously and emits its own completion event. Compensating transactions handle failures. This decoupling allows the shipping module to be replaced with a new implementation without touching the billing code, and the system can handle peak loads by buffering events rather than rejecting orders.
