---
title: Memory Swapping
description: The database server runs out of physical memory and starts using disk
  swap space, which dramatically slows down performance.
category:
- Performance
related_problems:
- slug: virtual-memory-thrashing
  similarity: 0.7
- slug: excessive-disk-io
  similarity: 0.7
- slug: high-database-resource-utilization
  similarity: 0.65
- slug: resource-contention
  similarity: 0.65
- slug: slow-application-performance
  similarity: 0.6
- slug: database-query-performance-issues
  similarity: 0.6
solutions:
- memory-management-optimization
- resource-usage-optimization
- resource-pooling
- backpressure
layout: problem
---

## Description
Memory swapping is a process where the operating system moves a block of memory (a "page") from RAM to disk to free up RAM for other processes. While this allows the system to continue functioning when it is low on memory, it comes at a significant performance cost, as accessing data from disk is much slower than accessing it from RAM. Frequent memory swapping is a strong indicator that a system does not have enough physical memory for its workload, and it can lead to a dramatic decrease in application performance.

## Indicators ⟡
- The server is slow, even when there are no obvious signs of high CPU usage.
- The server is using a lot of disk I/O, even when there is no heavy database load.
- The server is unresponsive or sluggish.
- You are getting complaints from users about slow performance.

## Symptoms ▲

- [Slow Application Performance](slow-application-performance.md)
<br/>  Disk-based swap is orders of magnitude slower than RAM access, causing dramatic application slowdowns.
- [Excessive Disk I/O](excessive-disk-io.md)
<br/>  Memory swapping generates significant disk I/O as pages are moved between RAM and disk swap space.
- [Database Query Performance Issues](database-query-performance-issues.md)
<br/>  When database server memory is swapped to disk, query execution becomes extremely slow as data must be read from disk instead of memory.
- [Service Timeouts](service-timeouts.md)
<br/>  The dramatic slowdown from swapping causes services to exceed their timeout thresholds, leading to cascading failures.
## Causes ▼

- [Memory Leaks](memory-leaks.md)
<br/>  Memory leaks gradually consume physical RAM until the system is forced to swap, as illustrated in the example of a Java application pushing MySQL into swapping.
- [High Database Resource Utilization](high-database-resource-utilization.md)
<br/>  Overloaded database servers consuming excessive memory push the system beyond physical RAM limits into swap territory.
- [Resource Contention](resource-contention.md)
<br/>  Multiple processes competing for limited physical memory force the OS to swap less active pages to disk.
## Detection Methods ○

- **System Monitoring Tools:** Use `free -h`, `vmstat`, `top`, or `htop` (Linux) to observe `swap` usage and `si`/`so` (swap in/out) rates.
- **Database Monitoring Tools:** Many database-specific monitoring tools will report memory usage and swap activity.
- **Cloud Provider Metrics:** If using a cloud-managed database, check the cloud provider's metrics for swap usage.
- **Alerting:** Set up alerts for high swap usage or high I/O wait times on database servers.

## Examples
A PostgreSQL database server, initially provisioned with 8GB of RAM, starts experiencing severe slowdowns after a year of operation. Investigation reveals that the `shared_buffers` setting was increased to 6GB, and the `work_mem` for many concurrent queries now exceeds the remaining physical memory, forcing the system to swap heavily. In another case, a Java application running on the same server as a MySQL database has a memory leak. Over several days, the Java application consumes more and more RAM, eventually pushing the MySQL database into heavy swapping, leading to application outages. This problem is particularly insidious because it can develop gradually as data volumes grow or application usage increases. It often indicates a fundamental resource bottleneck that needs to be addressed by adding more RAM, optimizing database configuration, or reducing memory consumption.
