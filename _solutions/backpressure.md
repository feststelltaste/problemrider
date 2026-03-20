---
title: Backpressure
description: Signaling producers to slow down when consumers become overwhelmed
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/backpressure/
problems:
- growing-task-queues
- task-queues-backing-up
- work-queue-buildup
- insufficient-worker-capacity
- thread-pool-exhaustion
- resource-contention
- virtual-memory-thrashing
- memory-swapping
- high-connection-count
- cascade-failures
layout: solution
---

## How to Apply ◆

> Legacy systems frequently lack any mechanism for downstream components to signal upstream producers that they are overwhelmed. Without backpressure, producers continue generating work at full speed while consumers fall further behind, leading to queue buildup, memory exhaustion, and cascading failures. Introducing backpressure creates a feedback loop that keeps the system in a sustainable operating range.

- Identify all producer-consumer boundaries in the system: message queues, API endpoints feeding background processors, batch job pipelines, and any point where work is generated faster than it can be consumed. These boundaries are where backpressure mechanisms must be applied.
- Introduce bounded queues at every producer-consumer boundary. Replace unbounded queues (which grow until memory is exhausted) with queues that have explicit maximum sizes. When a queue reaches capacity, the producer must either block, drop the message, or receive an error — all of which are preferable to silent, unbounded growth.
- Implement rate limiting on API endpoints and ingestion points to cap the rate at which new work enters the system. Use token bucket or sliding window algorithms to enforce sustainable throughput limits based on measured consumer capacity rather than arbitrary thresholds.
- Add queue depth monitoring with producer-side throttling: when queue depth exceeds a configurable threshold (typically 70-80% of capacity), reduce the producer's output rate. This can be implemented as a simple feedback loop where the producer polls queue depth or subscribes to queue metrics and adjusts its send rate accordingly.
- Use reactive streams or flow-control protocols (such as TCP flow control, gRPC flow control, Reactive Streams, or Kafka consumer group lag-based throttling) that have built-in backpressure semantics rather than implementing custom solutions. These protocols handle the complex coordination between producers and consumers automatically.
- Implement circuit breakers at service boundaries so that when a downstream service becomes overwhelmed, upstream callers receive fast failure responses instead of accumulating blocked threads waiting for the overloaded service. The circuit breaker trips when failure or latency thresholds are exceeded and resets after a cooldown period.
- Design batch processing systems to pull work rather than have work pushed to them. Pull-based architectures naturally implement backpressure because workers only request new tasks when they have capacity, preventing work from accumulating faster than it can be processed.
- Add load shedding as a last-resort backpressure mechanism: when the system is critically overloaded, selectively reject or deprioritize low-priority requests to preserve capacity for critical operations. Document which operations are eligible for shedding and communicate rejection clearly to callers so they can retry later.
- Test backpressure mechanisms under realistic overload conditions. Simulate scenarios where producers generate 2-5x the sustainable throughput and verify that the system degrades gracefully — slowing down or rejecting excess work — rather than consuming all available memory, crashing, or entering a thrashing state.

## Tradeoffs ⇄

> Backpressure prevents catastrophic failures under overload by keeping the system within its sustainable operating range, but it means that excess work is explicitly rejected, delayed, or throttled rather than silently accepted.

**Benefits:**

- Prevents queue buildup and eventual memory exhaustion by stopping unbounded work accumulation before it overwhelms system resources.
- Converts unpredictable system failures under overload into predictable, manageable degradation where excess requests receive clear rejection signals.
- Protects downstream services from being overwhelmed by upstream traffic spikes, preventing cascading failures across distributed systems.
- Enables the system to maintain consistent response times for accepted work even under heavy load, rather than degrading performance for all requests equally.
- Provides clear operational signals about system capacity limits, making it easier to determine when scaling is needed and by how much.

**Costs and Risks:**

- Callers must be designed to handle rejection or throttling signals, which requires changes to upstream systems that may not expect or handle pushback from downstream services.
- Misconfigured backpressure thresholds can reject work prematurely when there is still capacity available, reducing system throughput unnecessarily during normal operation.
- Bounded queues mean that during legitimate traffic spikes, some work may be rejected or delayed even though it would eventually be processed — this is a deliberate tradeoff of latency and availability over throughput.
- Implementing backpressure across service boundaries in a legacy system with many integration points requires coordinated changes across multiple components, which is difficult when different teams own different services.
- Load shedding decisions require clear business rules about which operations are expendable and which are critical, and getting these priorities wrong can cause more harm than the overload itself.

## How It Could Be

> The following scenarios illustrate how backpressure mechanisms prevent system failures in legacy systems under overload.

An order processing system uses a RabbitMQ queue to handle incoming orders from an e-commerce frontend. During flash sales, order volume spikes to 10x normal levels, and the queue grows to millions of messages, consuming all available memory on the message broker and causing it to crash. The team configures the queue with a maximum length of 50,000 messages and a dead-letter exchange for overflow. When the queue reaches capacity, new messages are routed to the dead-letter exchange where they are persisted to disk and retried during off-peak hours. Additionally, the frontend implements client-side rate limiting that displays a "high demand" waiting room when the API returns 429 (Too Many Requests) responses. During the next flash sale, the queue stays within bounds, the broker remains stable, and all orders are eventually processed — high-priority loyalty customers are processed immediately while overflow orders complete within 2 hours.

A data ingestion pipeline receives sensor data from 10,000 IoT devices and processes it through a series of transformation stages. As the device fleet grew, the second stage (data enrichment) became a bottleneck, causing the first stage to buffer increasingly large amounts of raw data in memory. Eventually, the buffering exhausted available RAM and triggered virtual memory thrashing that made the entire pipeline unusable. The team redesigned the pipeline using a pull-based architecture where each stage requests work from the previous stage only when it has processing capacity. When the enrichment stage falls behind, the ingestion stage automatically slows its acceptance of new sensor data, sending backpressure signals to the IoT gateway. The gateway responds by increasing its batching interval, reducing the per-second message rate while ensuring no data is lost. Memory usage stabilized at 2GB instead of growing unboundedly, and the pipeline handles sustained overload without degradation by smoothly throttling the intake rate.

A legacy banking application processes wire transfers through a queue-based workflow. During month-end, corporate clients submit thousands of bulk transfer files simultaneously, overwhelming the transfer processing workers. Without backpressure, the queue grows to 500,000 pending transfers, and workers begin thrashing as they compete for database connections and external validation service capacity. The team implements a tiered backpressure system: the API gateway enforces a per-client submission rate of 100 transfers per minute, the queue is bounded at 10,000 entries with overflow routed to a secondary persistent store, and workers implement circuit breakers on the external validation service with a 5-second timeout and 30-second cooldown. During the next month-end peak, the system processes transfers at a steady rate of 2,000 per minute, clients receive clear feedback about submission rate limits, and all transfers complete within the same business day — compared to the previous month where the system crashed and required 3 days of manual recovery.
