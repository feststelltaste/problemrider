---
title: Asynchronous Processing
description: Decoupling of calls and execution through asynchronicity
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/asynchronous-processing
problems:
- slow-application-performance
- thread-pool-exhaustion
- slow-response-times-for-lists
- growing-task-queues
- task-queues-backing-up
- external-service-delays
- cascade-failures
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify operations that do not need to complete before returning a response to the user: email sending, report generation, audit logging
- Introduce message queues or event buses to decouple request handling from long-running processing
- Convert synchronous blocking calls to external services into asynchronous operations with callbacks or futures
- Implement proper error handling for asynchronous workflows including retry logic and dead letter queues
- Use async/await patterns or reactive programming where the platform supports them
- Ensure idempotency in asynchronous handlers so that retried messages do not cause duplicate effects
- Monitor queue depths and processing latencies to detect bottlenecks early

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Improves responsiveness by returning control to the caller immediately
- Increases throughput by allowing the system to process multiple operations concurrently
- Provides natural resilience against slow downstream services
- Enables better resource utilization by avoiding idle thread blocking

**Costs and Risks:**
- Increases system complexity with additional infrastructure (queues, workers)
- Debugging asynchronous workflows is harder than following synchronous call stacks
- Eventual consistency may surprise users who expect immediate results
- Error handling and retry logic require careful design to avoid data corruption
- Legacy code tightly coupled to synchronous patterns may require significant refactoring

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy order management system processed each order synchronously, including inventory reservation, payment processing, and shipping label generation. During peak sales events, response times exceeded 30 seconds as the system waited for each external service call to complete. The team refactored the workflow to accept the order synchronously (validating basic data and returning an order ID) and then process the remaining steps asynchronously via a message queue. Order placement response time dropped to under 500 milliseconds, and the system handled three times the previous peak load because slow downstream services no longer blocked the request-handling threads.
