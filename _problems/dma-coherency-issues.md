---
title: DMA Coherency Issues
description: Direct Memory Access operations conflict with CPU cache coherency, leading
  to data corruption or inconsistent data views between CPU and DMA devices.
category:
- Code
- Data
- Performance
related_problems:
- slug: false-sharing
  similarity: 0.55
layout: problem
---

## Description

DMA coherency issues occur when Direct Memory Access devices and the CPU have different views of the same memory data due to cache coherency problems. DMA devices can read and write memory directly without going through the CPU cache, while the CPU may have cached copies of the same data. This can lead to data corruption, lost updates, or inconsistent system behavior when cached and non-cached views of memory diverge.

## Indicators ⟡

- Data corruption occurs intermittently in DMA-based operations
- System behavior varies based on CPU cache state or timing
- Network or disk I/O operations show data inconsistency
- Performance issues related to excessive cache flushing or invalidation
- Problems appear more frequently under high system load or specific timing conditions

## Symptoms ▲

- [Silent Data Corruption](silent-data-corruption.md)
<br/>  When DMA and CPU cache views diverge, data can be silently corrupted without triggering errors, as the system processes stale or inconsistent memory contents.
- [Race Conditions](race-conditions.md)
<br/>  DMA coherency problems manifest as race conditions between the CPU cache and DMA device accessing the same memory regions concurrently.
- [Debugging Difficulties](debugging-difficulties.md)
<br/>  DMA coherency issues are timing-dependent and may not reproduce consistently, making them extremely difficult to diagnose and debug.
- [Gradual Performance Degradation](gradual-performance-degradation.md)
<br/>  Workarounds like excessive cache flushing or invalidation to address coherency issues progressively degrade system performance.
- [Increased Error Rates](increased-error-rates.md)
<br/>  Inconsistent memory views between CPU and DMA devices lead to sporadic errors in I/O operations, network processing, and data transfers.
## Causes ▼

- [False Sharing](false-sharing.md)
<br/>  When DMA buffers share cache lines with CPU-accessed data, false sharing creates coherency conflicts between CPU cache and DMA operations.
- [Poor Caching Strategy](poor-caching-strategy.md)
<br/>  Failure to properly manage cache coherency for DMA-accessible memory regions, such as not using uncacheable mappings or proper flush/invalidate operations, leads to coherency issues.
- [Alignment and Padding Issues](alignment-and-padding-issues.md)
<br/>  Poor memory alignment of DMA buffers can cause them to share cache lines with non-DMA data, creating coherency conflicts.
## Detection Methods ○

- **DMA Operation Monitoring:** Monitor DMA transfers and their interaction with CPU cache
- **Data Integrity Verification:** Compare expected vs actual data after DMA operations
- **Cache Coherency Testing:** Test under different cache states and loading conditions
- **Hardware Performance Monitoring:** Use hardware counters to detect coherency issues
- **Memory Access Pattern Analysis:** Analyze patterns of CPU and DMA memory access
- **Platform-Specific Testing:** Test on different hardware platforms with varying coherency models

## Examples

A network card receives packets via DMA into system memory buffers that the CPU has previously cached. The CPU reads packet headers from its cache while the DMA operation overwrites the same memory with new packet data. The CPU processes stale cached header information while the actual packet data in memory is different, leading to incorrect packet processing and network protocol violations. Another example involves a graphics driver that uses DMA to transfer vertex data to a GPU while the CPU simultaneously updates the same vertex buffer. Without proper cache synchronization, the GPU receives partially cached and partially updated vertex data, causing rendering artifacts and corrupted 3D models.
