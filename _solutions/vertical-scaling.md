---
title: Vertical Scaling
description: Increasing the performance of individual components
category:
- Performance
- Operations
problems:
- slow-application-performance
- capacity-mismatch
- scaling-inefficiencies
- slow-database-queries
- high-database-resource-utilization
- gradual-performance-degradation
layout: solution
related_solutions:
- slug: horizontal-scaling
  similarity: 0.85
- slug: distributed-caching
  similarity: 0.75
- slug: data-replication
  similarity: 0.7
- slug: load-balancing
  similarity: 0.7
- slug: specialized-hardware
  similarity: 0.7
- slug: denormalization
  similarity: 0.7
---

## Description

Vertical scaling increases the capacity of a single component — more CPU cores, more memory, faster storage — rather than distributing load across additional instances, and it requires no changes to application code, which makes it the fastest available lever for relieving a performance bottleneck. Its appeal in legacy contexts is precisely that it can be applied without touching software that nobody fully understands anymore: a legacy application whose architecture assumes a single-instance deployment, or that simply cannot be safely refactored for horizontal scaling on any reasonable timeline, can often still be given meaningfully more headroom just by upgrading the hardware or infrastructure it already runs on. The mechanism only pays off, however, if the actual bottleneck is correctly diagnosed first — CPU, memory, I/O, or network — since throwing additional resources at a component that isn't the real constraint accomplishes nothing. Because vertical scaling has a hard ceiling determined by available hardware and a non-linear cost curve as instances get larger, it functions best as a deliberate short-term measure that buys time and breathing room for a legacy system under acute performance pressure, rather than as a permanent substitute for addressing the architectural bottlenecks that limit any single instance's capacity. Used this way, it converts an urgent capacity crisis into a manageable one, giving the team the room to plan a more structural fix — partitioning, horizontal scaling, or an architecture change — without that planning happening under emergency conditions.

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
