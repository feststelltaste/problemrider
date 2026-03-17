---
title: Resource Contention
description: The server is overloaded, and the application is competing for limited
  resources like CPU, memory, or I/O.
category:
- Code
- Performance
related_problems:
- slug: high-client-side-resource-consumption
  similarity: 0.7
- slug: high-database-resource-utilization
  similarity: 0.7
- slug: high-resource-utilization-on-client
  similarity: 0.7
- slug: excessive-disk-io
  similarity: 0.65
- slug: memory-swapping
  similarity: 0.65
- slug: lock-contention
  similarity: 0.65
layout: problem
---

## Description
Resource contention occurs when multiple processes or threads compete for the same limited resources, such as CPU, memory, or I/O. This competition can lead to performance degradation, as processes are forced to wait for resources to become available. In severe cases, it can lead to deadlocks or other forms of system instability. Understanding and managing resource contention is a key aspect of building scalable and performant systems.

## Indicators ⟡
- The server is slow, even when there are no obvious signs of high CPU usage.
- The server is using a lot of disk I/O, even when there is no heavy database load.
- The server is unresponsive or sluggish.
- You are getting complaints from users about slow performance.

## Symptoms ▲

- [Slow Response Times for Lists](slow-response-times-for-lists.md)
<br/>  Resource contention causes data-intensive operations like list queries to slow down significantly as processes compete for I/O and CPU.
- [Gradual Performance Degradation](gradual-performance-degradation.md)
<br/>  As resource competition intensifies over time, overall system performance steadily deteriorates.
- [Cascade Failures](cascade-failures.md)
<br/>  When resources are exhausted, components begin failing in sequence as they cannot obtain the resources they need to function.
- [Memory Swapping](memory-swapping.md)
<br/>  Heavy memory contention forces the OS to swap memory to disk, dramatically degrading system performance.
- [Unpredictable System Behavior](unpredictable-system-behavior.md)
<br/>  Resource contention causes timing-dependent behavior where system performance varies unpredictably based on concurrent load patterns.
## Causes ▼

- [Resource Allocation Failures](resource-allocation-failures.md)
<br/>  Leaked resources reduce available capacity, intensifying competition among processes for the remaining resources.
- [N+1 Query Problem](n-plus-one-query-problem.md)
<br/>  Excessive database queries from N+1 patterns consume database resources and create I/O contention.
- [Capacity Mismatch](capacity-mismatch.md)
<br/>  Infrastructure that doesn't match actual demand patterns leads to resource contention during peak usage periods.
- [Scaling Inefficiencies](scaling-inefficiencies.md)
<br/>  Inability to scale components independently means bottlenecked components create resource contention for the entire system.
## Detection Methods ○

- **System Monitoring Tools:** Use tools like `top`, `htop`, `vmstat`, `iostat` (Linux) or Task Manager (Windows) to monitor CPU, memory, and I/O usage.
- **Application Performance Monitoring (APM):** APM tools can often show resource utilization at the application level and help pinpoint which parts of the application are resource-intensive.
- **Load Testing:** Simulate high load to identify resource bottlenecks and contention points.
- **Profiling:** Use profiling tools to identify code sections that are consuming excessive CPU or memory.

## Examples
A web server experiences slow response times during peak hours. Monitoring reveals that the CPU utilization is consistently at 100%. This indicates that the server does not have enough CPU capacity to handle the incoming requests. In another case, a database server is experiencing high I/O wait times. Investigation reveals that multiple applications are performing large, unindexed queries simultaneously, leading to disk contention. This problem is common in systems that are not properly scaled or where resource usage patterns have changed over time. It often requires a combination of capacity planning, code optimization, and infrastructure tuning to resolve.
