---
title: Saga Pattern
description: Managing distributed transactions through sequences of local transactions with compensating actions
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/reliability/saga-pattern
problems:
- cascade-failures
- long-running-transactions
- tight-coupling-issues
- unpredictable-system-behavior
- microservice-communication-overhead
- data-migration-integrity-issues
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify distributed transactions in the legacy system that span multiple services or databases
- Decompose each distributed transaction into a sequence of local transactions with defined ordering
- Design compensating actions for each step that can undo its effects if a subsequent step fails
- Choose between choreography (event-driven) and orchestration (central coordinator) based on system complexity
- Implement idempotent operations at each step to handle retries safely
- Add monitoring and alerting for sagas that remain in intermediate states beyond expected durations
- Store saga state persistently to survive process restarts and enable recovery of in-progress sagas

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates the need for distributed two-phase commit, which is fragile in legacy environments
- Enables data consistency across service boundaries without tight coupling
- Each local transaction can use the most appropriate data store and isolation level
- Failed transactions are automatically compensated rather than left in inconsistent states

**Costs and Risks:**
- Compensating actions add significant design and implementation complexity
- Temporary data inconsistency is visible between saga steps (eventual consistency)
- Debugging failed sagas across multiple services is challenging
- Some operations are difficult or impossible to compensate (emails sent, physical goods shipped)

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A travel agency's legacy booking system used a single database transaction to reserve flights, hotels, and car rentals simultaneously. As the system was decomposed into separate services, the monolithic transaction broke. The team implemented a saga pattern where each booking step was a local transaction with a compensating cancellation action. If the hotel reservation succeeded but the car rental failed, the saga automatically canceled the hotel reservation and notified the flight service to release the seat. An orchestrator service tracked saga state and handled retries for transient failures, providing the same all-or-nothing booking semantics without distributed transactions.
