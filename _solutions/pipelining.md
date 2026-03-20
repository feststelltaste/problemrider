---
title: Pipelining
description: Simultaneous execution of sequential processing steps
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/pipelining
problems:
- slow-application-performance
- bottleneck-formation
- growing-task-queues
- long-build-and-test-times
- scaling-inefficiencies
- work-queue-buildup
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify sequential processing workflows where one stage's output feeds into the next (e.g., extract-transform-load, request processing chains)
- Decompose the workflow into discrete stages that can operate on different data items concurrently
- Connect stages with bounded queues or channels to manage backpressure and prevent memory exhaustion
- Ensure each stage is independently scalable so bottleneck stages can be given more resources
- Implement monitoring for each stage's throughput and queue depth to identify which stages limit overall throughput
- Start by pipelining the most sequential and time-consuming workflows, then expand to other areas

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Increases overall throughput by overlapping the execution of sequential steps
- Makes bottleneck stages visible and independently addressable
- Improves resource utilization by keeping all processing units busy simultaneously
- Enables streaming processing of large datasets without loading everything into memory

**Costs and Risks:**
- Adds complexity in error handling when failures occur mid-pipeline
- Debugging issues becomes harder when data flows through multiple concurrent stages
- Backpressure management is critical; without it, fast producers can overwhelm slow consumers
- Legacy code with tightly coupled sequential steps requires significant refactoring to pipeline
- Increases latency for individual items even as throughput increases

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A data warehousing system processed nightly data feeds by sequentially extracting data from source systems, transforming it through business rules, and loading it into the warehouse. Each phase waited for the previous one to complete fully, resulting in a 10-hour processing window. The team restructured the pipeline so that extraction, transformation, and loading operated concurrently on different data batches. As soon as the first batch was extracted, transformation began on it while extraction continued on the next batch. This overlap reduced total processing time to under 4 hours, comfortably fitting within the overnight window even as data volumes continued to grow.
