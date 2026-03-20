---
title: Dead Letter Queue
description: Routing failed messages to a dedicated queue for later inspection and reprocessing instead of losing them
category:
- Architecture
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/dead-letter-queue
problems:
- silent-data-corruption
- inadequate-error-handling
- cascade-failures
- monitoring-gaps
- task-queues-backing-up
- increased-error-rates
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify all message queues and asynchronous processing pipelines in the legacy system
- Configure dead letter queues for each processing queue, routing messages that fail after a defined number of retries
- Include the original message payload, error details, retry count, and timestamp in dead letter entries
- Build monitoring and alerting on dead letter queue depth to detect processing failures early
- Create tooling for inspecting dead letter messages, diagnosing failures, and replaying them after fixes
- Define retention policies for dead letter messages based on regulatory and business requirements
- Implement automated classification of dead letter messages to identify recurring failure patterns

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Prevents data loss from transient failures or processing errors
- Provides a diagnostic tool for understanding why messages fail
- Decouples error handling from the main processing pipeline, keeping it clean
- Enables message replay after bug fixes without re-triggering upstream systems
- Prevents poison messages from blocking the main processing queue

**Costs and Risks:**
- Dead letter queues require monitoring; unattended queues can grow unboundedly
- Replayed messages may cause side effects if the system is not idempotent
- Adds infrastructure and operational complexity for each message queue
- Stale dead letter messages may become invalid if the system has changed since the original failure

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy order fulfillment system processed messages from a RabbitMQ queue. When a message could not be processed due to data validation errors or downstream service failures, the message was discarded with only a log entry. During a payment gateway outage, 2,400 orders were lost permanently. After this incident, the team added dead letter queues to all processing stages. Failed messages were routed to DLQs with full error context. A simple web dashboard allowed operators to inspect, filter, and replay dead letter messages. When a similar payment gateway issue occurred three months later, all 1,800 affected orders were automatically captured in the DLQ and successfully reprocessed once the gateway recovered.
