---
title: Capacity Planning
description: Estimating future resource needs from growth projections and performance models
category:
- Performance
- Operations
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/capacity-planning/
problems:
- capacity-mismatch
- scaling-inefficiencies
- growing-task-queues
- task-queues-backing-up
- work-queue-buildup
- insufficient-worker-capacity
- thread-pool-exhaustion
- high-connection-count
layout: solution
---

## How to Apply ◆

> Legacy systems often run without any formal understanding of their resource limits or growth trajectories. Capacity planning introduces a disciplined approach to forecasting demand and ensuring the system can meet it before failures occur.

- Establish a baseline by measuring the current resource consumption of the legacy system under normal and peak conditions. Capture CPU, memory, disk I/O, network bandwidth, database connections, thread pool utilization, and queue depths. Without an accurate baseline, all future projections are guesswork.
- Correlate resource consumption with business metrics such as active users, transactions per hour, or data volume ingested. This correlation model lets you translate business growth forecasts into concrete resource requirements rather than relying on arbitrary multipliers.
- Identify the system's current saturation points by load testing or analyzing historical incidents. Determine which resource exhausts first under increasing load — this is the binding constraint that will trigger failures. In legacy systems, these constraints are often database connections, thread pools, or memory, not CPU.
- Build a simple capacity model that maps projected workload growth to resource demand. Even a spreadsheet-based model that extrapolates current trends and applies known per-transaction costs provides far more insight than no model at all. Update the model quarterly with fresh measurements.
- Define capacity thresholds and alert boundaries at 70% and 85% utilization for each critical resource. These thresholds give operations teams lead time to act before saturation causes user-visible problems. In legacy systems where scaling is slow or manual, earlier warning thresholds are essential.
- Plan for peak periods explicitly by analyzing historical patterns such as end-of-month processing, seasonal traffic spikes, or batch job scheduling conflicts. Legacy systems frequently have batch workloads that compete with interactive traffic, and capacity plans must account for both running simultaneously.
- Size worker pools, thread pools, and connection pools based on measured per-request resource costs and projected concurrency, not on defaults or values inherited from initial deployment. Document the rationale behind each pool size so that future maintainers can adjust settings as workloads change.
- Incorporate capacity planning into the change management process. Before deploying new features or integrations to a legacy system, estimate their resource impact and verify that current capacity can absorb it. Many legacy system outages stem from new workloads added without considering their effect on already-constrained resources.

## Tradeoffs ⇄

> Capacity planning reduces the risk of outages and performance degradation by anticipating resource needs, but it requires ongoing investment in measurement, modeling, and organizational discipline.

**Benefits:**

- Prevents resource exhaustion failures by providing early visibility into approaching limits for database connections, thread pools, worker processes, and queue capacity.
- Reduces emergency scaling incidents by giving teams weeks or months of lead time rather than hours, which is especially valuable for legacy systems where scaling involves procurement, configuration, or architectural changes that cannot be done quickly.
- Enables informed decisions about hardware investments, cloud resource provisioning, and architectural refactoring by grounding them in measured data rather than intuition.
- Improves reliability during peak periods by ensuring the system is provisioned for known demand spikes rather than being caught off guard by predictable seasonal or batch workload increases.
- Creates institutional knowledge about system limits and growth patterns that persists even when team members change, which is critical for legacy systems with limited documentation.

**Costs and Risks:**

- Requires instrumentation and monitoring that may not exist in the legacy system, and adding it can be costly in codebases that were not designed for observability.
- Capacity models based on current architecture may become invalid if the system is refactored, migrated, or significantly changed, requiring the model to be rebuilt.
- Overreliance on projections can lead to over-provisioning, wasting budget on resources that are never used, particularly when growth estimates are optimistic.
- Maintaining and updating the capacity model requires ongoing effort from engineers who understand both the system internals and the business context, which competes with feature development time.
- In legacy systems with opaque or poorly understood resource consumption patterns, building an accurate model may require significant upfront investigation and profiling effort.

## Examples

> The following scenarios illustrate how capacity planning addresses resource management problems in legacy systems.

A financial services company runs a 15-year-old transaction processing system that handles payment settlements. The system uses a fixed pool of 50 database connections and 32 worker threads, values that were set at initial deployment and never revisited. Transaction volume has grown 8% year-over-year, and the team begins experiencing intermittent connection exhaustion during end-of-month settlement runs. By establishing a capacity model that correlates transaction volume with connection usage and thread utilization, the team determines they will permanently exceed their connection pool capacity within six months at current growth rates. They proactively increase the pool sizes, add connection pooling via pgBouncer, and schedule batch settlements to avoid overlap with peak interactive traffic. The monthly outages stop, and the capacity model becomes a standing agenda item in quarterly operations reviews.

A logistics platform processes shipment tracking events through a message queue with a fixed set of worker processes. Over two years, the number of tracked shipments doubles, but the worker count remains unchanged. Task queues begin backing up during peak shipping hours, causing tracking updates to be delayed by several hours. After instrumenting the system to measure per-event processing cost and correlating queue depth with shipment volume, the team builds a projection showing that current worker capacity will be fully saturated within three months. They implement auto-scaling rules for worker processes tied to queue depth thresholds and establish monthly capacity reviews that compare actual growth against projections. Queue backup incidents drop from weekly occurrences to near zero, and the team gains confidence in planning infrastructure budgets based on measured data rather than reactive firefighting.
