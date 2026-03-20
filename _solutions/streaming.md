---
title: Streaming
description: Continuous processing and transmission of data
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/streaming
problems:
- slow-application-performance
- unbounded-data-growth
- growing-task-queues
- bottleneck-formation
- scaling-inefficiencies
- work-queue-buildup
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify batch-oriented processes that could benefit from continuous processing (e.g., data ingestion, event processing, ETL pipelines)
- Introduce a streaming platform (Kafka, RabbitMQ Streams, Kinesis) as the backbone for real-time data flow
- Refactor batch file-transfer integrations into event streams, producing events as they occur rather than accumulating them
- Implement stream processing for legacy reporting that currently depends on end-of-day batch runs
- Use change data capture (CDC) to stream database changes from legacy systems without modifying their code
- Apply windowing and aggregation at the stream level to produce near-real-time analytics

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables real-time data processing and reduces end-to-end latency from hours to seconds
- Handles continuous data flows without accumulating large intermediate datasets
- Decouples producers and consumers, allowing independent scaling and evolution
- Naturally handles backpressure through consumer group management

**Costs and Risks:**
- Streaming infrastructure adds operational complexity compared to simple batch processing
- Exactly-once processing semantics are difficult to achieve and verify
- Debugging streaming pipelines requires specialized tooling and expertise
- Legacy systems designed around batch paradigms may need significant refactoring to produce events
- Ordering guarantees across partitions require careful design

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A logistics company's legacy system accumulated shipment tracking events in a staging table and processed them in hourly batch runs. As shipment volumes grew, the hourly batch took longer than an hour to complete, causing a growing backlog. The team implemented Kafka-based event streaming with CDC on the legacy database, processing tracking events as they arrived. This eliminated the batch backlog, reduced tracking update latency from up to two hours to under 10 seconds, and enabled real-time customer notifications that had previously been impossible.
