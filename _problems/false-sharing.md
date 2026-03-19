---
title: False Sharing
description: Multiple CPU cores access different variables located on the same cache
  line, causing unnecessary cache coherency traffic and performance degradation.
category:
- Code
- Performance
related_problems:
- slug: race-conditions
  similarity: 0.6
- slug: memory-barrier-inefficiency
  similarity: 0.55
- slug: atomic-operation-overhead
  similarity: 0.55
- slug: lock-contention
  similarity: 0.55
- slug: dma-coherency-issues
  similarity: 0.55
- slug: data-structure-cache-inefficiency
  similarity: 0.5
layout: problem
---

## Description

False sharing occurs when multiple CPU cores access different data elements that happen to reside on the same cache line, causing the cache coherency protocol to invalidate and transfer cache lines between cores even though the cores are not actually sharing data logically. This creates unnecessary memory traffic and performance degradation in multi-threaded applications, as cores compete for cache lines containing unrelated data.

## Indicators ⟡

- Multi-threaded performance scales poorly with increased thread count
- Cache coherency traffic is high relative to actual data sharing requirements
- Performance degrades when threads access seemingly independent data structures
- Profiling shows excessive cache line transfers between CPU cores
- Single-threaded performance is good but multi-threaded performance is poor

## Symptoms ▲

- [Slow Application Performance](slow-application-performance.md)
<br/>  False sharing causes unnecessary cache coherency traffic that degrades multi-threaded application performance, making the application noticeably slower.
- [Resource Contention](resource-contention.md)
<br/>  Cores compete for cache lines containing unrelated data, creating artificial resource contention at the hardware level.
- [Scaling Inefficiencies](scaling-inefficiencies.md)
<br/>  False sharing prevents linear performance scaling with additional threads or cores, as adding more parallelism increases cache coherency overhead.
- [Gradual Performance Degradation](gradual-performance-degradation.md)
<br/>  As more threads are added over time, false sharing effects compound, causing progressively worse performance degradation.

## Causes ▼

- [Alignment and Padding Issues](alignment-and-padding-issues.md)
<br/>  Poor data structure alignment places independent variables on the same cache line, directly causing false sharing between cores.
- [Data Structure Cache Inefficiency](data-structure-cache-inefficiency.md)
<br/>  Data structures organized without consideration for cache line boundaries lead to unrelated data sharing cache lines.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers without knowledge of CPU cache architecture may inadvertently create data layouts that cause false sharing.
## Detection Methods ○

- **Cache Performance Profiling:** Use profilers that can detect cache line contention and false sharing
- **Hardware Performance Counters:** Monitor cache coherency events and inter-core traffic
- **Memory Layout Analysis:** Examine data structure layouts and memory alignment
- **Thread Affinity Testing:** Test performance with different thread-to-core assignments
- **Padding Experiments:** Add padding between data structures to test for false sharing effects
- **Cache Line Analysis Tools:** Use specialized tools that detect false sharing patterns

## Examples

A multi-threaded counter application has an array of counter variables, one per thread, to avoid synchronization. However, if multiple counters fit on the same 64-byte cache line, threads updating different counters cause cache line ping-ponging between CPU cores. Each update by one thread invalidates the cache line for other threads, forcing them to reload the entire cache line even though they're updating completely different counters. Another example involves a producer-consumer system where producer and consumer threads each have their own index variables (head and tail) for a circular buffer. If these indices are stored adjacently in memory, updating one index causes the other thread's cache line to be invalidated, creating false contention for data that is logically independent.
