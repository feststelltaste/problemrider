---
title: Fault-Tolerant Data Structures
description: Use of data structures that remain operational despite errors or inconsistencies
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/reliability/fault-tolerant-data-structures
problems:
- silent-data-corruption
- inadequate-error-handling
- unpredictable-system-behavior
- brittle-codebase
- data-migration-integrity-issues
- cascade-failures
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Audit critical data structures in the legacy codebase for vulnerability to corruption or partial writes
- Introduce checksums or version fields in data records to detect inconsistencies early
- Use self-healing data structures such as redundant linked lists or B-trees with integrity verification
- Implement defensive deserialization that validates structural invariants before accepting data
- Add recovery logic that can rebuild or repair data structures from known-good state or logs
- Wrap legacy data access behind validation layers that reject or quarantine corrupted entries

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- System continues operating even when individual data elements are corrupted
- Reduces silent data corruption that can propagate through downstream processes
- Makes data problems visible through integrity checks rather than mysterious failures
- Supports safer data migration by detecting inconsistencies during transition

**Costs and Risks:**
- Fault-tolerant structures use more memory and CPU for redundancy and validation
- Retrofitting existing data formats requires careful migration planning
- Over-reliance on self-healing can mask systemic problems that need root cause fixes
- Added complexity in data access layers increases maintenance burden

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A telecommunications provider discovered that its legacy billing system occasionally produced corrupted customer records due to race conditions in a shared-memory data structure. By replacing the critical account balance cache with a versioned structure that included CRC checks and automatic rollback to the last valid state, the team eliminated billing discrepancies that had been causing customer complaints for years. The fault-tolerant structure logged every detected corruption event, which also helped the team identify and fix the underlying concurrency bug.
