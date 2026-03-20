---
title: Vertical Scaling
description: Increasing the performance of individual components
category:
- Performance
- Operations
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/vertical-scaling
problems:
- slow-application-performance
- capacity-mismatch
- scaling-inefficiencies
- slow-database-queries
- high-database-resource-utilization
- gradual-performance-degradation
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Profile the system to determine whether the bottleneck is CPU, memory, I/O, or network before upgrading hardware
- Increase server resources (CPU cores, RAM, faster storage) for the component identified as the constraint
- Upgrade database servers with more memory to keep working sets cached and reduce disk I/O
- Replace HDD with SSD or NVMe storage for I/O-bound legacy applications and databases
- Tune application and database server configurations to take advantage of additional resources (thread pools, buffer pools, heap sizes)
- Use vertical scaling as a short-term measure to buy time while planning horizontal scaling or architectural improvements
- Document the scaling ceiling for the current architecture so the team knows when vertical scaling will no longer suffice

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Simplest scaling approach that requires no application code changes
- Immediately effective for legacy applications that cannot be horizontally scaled
- Maintains the existing single-instance deployment model, avoiding distributed system complexity
- Often the fastest path to resolving an acute performance crisis

**Costs and Risks:**
- Hard ceiling on vertical scaling determined by available hardware
- Larger instances are disproportionately expensive (non-linear cost curve)
- Does not address architectural bottlenecks that limit single-instance performance
- Can mask underlying problems, delaying necessary refactoring
- Creates a single point of failure with higher blast radius as more load concentrates on one machine

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy accounting system running on a server with 16 GB of RAM and spinning disks experienced severe performance degradation as the transaction database grew beyond 500 GB. Analysis showed that the database buffer pool could only cache 20 percent of the working set, causing constant disk I/O. The team upgraded the server to 128 GB of RAM and NVMe storage. Database query times improved by 10x, and the end-of-month close process that had stretched to 14 hours completed in 90 minutes. The team used the performance breathing room to plan a database partitioning strategy for when the dataset would exceed even the upgraded server's capacity.
