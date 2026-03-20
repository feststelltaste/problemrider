---
title: Event-Driven Architecture
description: Decoupling components through asynchronous events for independent evolution and modification
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/event-driven-architecture/
problems:
- tight-coupling-issues
- deployment-coupling
- high-coupling-and-low-cohesion
- cascade-failures
- monolithic-architecture-constraints
- single-points-of-failure
- circular-dependency-problems
- bottleneck-formation
- load-balancing-problems
- service-timeouts
- upstream-timeouts
layout: solution
---

## How to Apply ◆

> In a legacy context, event-driven architecture is most valuable as a decoupling mechanism that lets new and old components evolve independently without requiring simultaneous changes across tightly woven call chains.

- Map the synchronous call chains in the existing legacy system and identify the links where one component blocks another unnecessarily — these are the first candidates for replacement with asynchronous events.
- Introduce an event broker (such as Kafka or RabbitMQ) as a seam between the legacy system and new components, so that legacy components can publish events without knowing who consumes them.
- Define domain events that capture meaningful business facts — "order placed," "payment received," "inventory reserved" — using language from the business domain rather than the legacy system's internal terminology, and treat these events as the stable public contract between old and new parts.
- Design all new event consumers to be idempotent from the start; legacy systems often have retry logic and batch replays that will deliver the same event more than once, and non-idempotent consumers will corrupt data silently.
- Use dead letter queues for every consumer and monitor their depth; legacy-produced events frequently contain unexpected formats or missing fields that will break consumers, and silent queue blocking is a common failure mode during modernization.
- Apply saga patterns to replace distributed transactions that span legacy and modern components; instead of a two-phase commit across an old RDBMS and a new service, model the workflow as a sequence of compensating events.
- Avoid event-driven communication for interactions that genuinely require immediate responses from the legacy back end — authorization checks and real-time inventory lookups belong on synchronous calls; background processing and notifications do not.
- Version event schemas explicitly from the beginning; once other teams or new services subscribe to an event stream, breaking its format is as disruptive as breaking a public REST API.

## Tradeoffs ⇄

> Event-driven architecture is a powerful tool for loosening the grip that a monolithic legacy system has on the components that surround it, but it shifts complexity from call-chain coupling to event flow management.

**Benefits:**

- Breaks synchronous call chains that cause cascade failures in the legacy system, so that a slow or unavailable legacy component no longer blocks the entire user-facing request path.
- Allows new services to be added to the ecosystem without modifying the legacy system — they simply subscribe to existing event topics.
- Absorbs load mismatches between a fast legacy producer and a slower modern consumer through the broker's buffer, reducing the need to throttle the legacy system.
- Creates a durable event log that can serve as the foundation for audit trails and data migration, both of which are frequently needed during legacy modernization.
- Enables independent deployment of new consumers without coordinating releases with the legacy system's release schedule.

**Costs and Risks:**

- Debugging is significantly harder when a business process spans a legacy publisher, a broker, and multiple modern consumers, because there is no single call stack to inspect.
- Eventual consistency introduces a window during which the legacy system and new services have different views of reality, which requires deliberate handling and is often unfamiliar to teams used to synchronous legacy patterns.
- The event broker itself becomes a new operational dependency that must be sized, monitored, and kept available — adding infrastructure burden at a time when teams are already managing legacy infrastructure.
- Legacy systems were rarely designed to publish events; retrofitting event publication often means adding outbox tables, log-based change data capture, or polling bridges, each of which carries its own fragility.
- Event schema changes in a legacy-produced feed are difficult to coordinate because the team that owns the legacy system may not be aware of all downstream consumers.

## How It Could Be

> The following scenarios show how event-driven architecture has been used to relieve the coupling pressure of legacy systems in real modernization programs.

A national insurance provider ran all of its claim intake, underwriting, and payment through a single monolithic J2EE application deployed as one WAR file. Adding a new notification channel or a fraud detection step required changes to the monolith's core orchestration layer, which triggered a full regression cycle and a quarterly release. The team introduced a Kafka broker alongside the monolith and instrumented the monolith's service layer to publish a `ClaimSubmitted` event after every successful intake. New microservices for fraud screening and customer notifications subscribed independently, each deployable on their own schedule. The monolith's release cycle did not change, but new capabilities could now be delivered weekly.

A manufacturing company integrated a thirty-year-old ERP system with a new warehouse management system. The ERP batch-exported inventory adjustment files every four hours to a shared network drive; the warehouse system polled the drive and imported the files. When file import failed, the warehouse team had no visibility until stock counts diverged enough to cause picking errors. The team replaced the file-poll mechanism with a change-data-capture bridge that read the ERP's database transaction log and published inventory events to RabbitMQ. The warehouse system consumed the queue and processed adjustments in near-real-time. Dead letter queue monitoring gave the operations team immediate visibility into import failures, replacing the previous four-hour blind spot with a minute-level alert.

A regional bank operated a loan servicing platform built in the early 2000s that handled repayment processing synchronously: incoming repayment triggered payment posting, statement generation, amortization schedule recalculation, and regulatory reporting all in a single database transaction. As transaction volumes grew, this chain routinely timed out, causing customer-facing errors and requiring manual reconciliation. The modernization team decomposed the chain into events: the legacy system posted the payment and published a `PaymentPosted` event. Statement generation, schedule recalculation, and regulatory reporting became independent consumers. The synchronous transaction shrank to the payment posting step alone, reducing timeout frequency dramatically, and the downstream processes could be scaled independently to handle peak quarter-end volume.
