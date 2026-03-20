---
title: Memory Hierarchy
description: Utilizing locality of memory accesses at different levels
category:
- Performance
- Code
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/memory-hierarchy
problems:
- slow-application-performance
- data-structure-cache-inefficiency
- memory-fragmentation
- excessive-object-allocation
- gradual-performance-degradation
- inefficient-code
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Profile memory access patterns using tools like perf, VTune, or cachegrind to identify cache-unfriendly code paths
- Reorganize data structures to improve spatial locality, favoring arrays of structs or struct-of-arrays layouts depending on access patterns
- Reduce pointer chasing by replacing linked structures with contiguous arrays where iteration dominates
- Align data structures to cache line boundaries to prevent false sharing in concurrent code
- Batch processing of data to operate on cache-resident subsets rather than streaming through entire datasets randomly
- Review hot loops in legacy code for unnecessary indirection layers that defeat prefetching

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Can yield dramatic speedups (2-10x) for data-intensive operations without algorithmic changes
- Reduces memory bandwidth pressure, benefiting the entire system
- Improvements are durable and do not degrade over time like cache-based solutions might

**Costs and Risks:**
- Requires deep understanding of hardware behavior that many application developers lack
- Optimized data layouts can be less readable and harder to maintain
- Changes to data structure layout can ripple through legacy codebases with tight coupling
- Benefits are hardware-dependent and may not transfer across different processor architectures
- Over-optimization can make code brittle and difficult to evolve

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A scientific computing application processed simulation data using a linked-list-based particle system that had been in place for over a decade. Profiling revealed that 60 percent of execution time was spent on cache misses during particle iteration. The team replaced the linked list with a contiguous array and reorganized the particle struct to place frequently accessed fields (position, velocity) adjacent in memory. The change reduced cache miss rates by 80 percent and cut overall simulation time nearly in half, with no change to the underlying algorithm.
