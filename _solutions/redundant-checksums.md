---
title: Redundant Checksums
description: Using multiple different checksum algorithms
category:
- Code
- Security
problems:
- silent-data-corruption
- data-migration-integrity-issues
- inadequate-error-handling
- unpredictable-system-behavior
- dma-coherency-issues
layout: solution
related_solutions:
- slug: checksums
  similarity: 0.9
- slug: error-correction-codes
  similarity: 0.8
- slug: redundant-data-storage
  similarity: 0.75
- slug: continuous-data-verification
  similarity: 0.75
- slug: fault-tolerant-data-structures
  similarity: 0.7
- slug: data-integrity
  similarity: 0.7
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify critical data flows in the legacy system where corruption would have severe business consequences
- Apply multiple independent checksum algorithms (e.g., CRC32 and SHA-256) to critical data in transit and at rest
- Verify checksums at every system boundary where data is received, stored, or transmitted
- Store checksums alongside the data they protect and validate them during read operations
- Implement automated alerting when checksum verification fails
- Use redundant checksums during data migration to verify that source and destination data match exactly

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Dramatically reduces the probability of undetected data corruption
- Provides strong guarantees for data integrity during migration and replication
- Catches corruption caused by hardware failures, software bugs, or transmission errors
- Two independent algorithms make it virtually impossible for corruption to pass both checks

**Costs and Risks:**
- Computational overhead of calculating multiple checksums on every data operation
- Additional storage required for multiple checksum values
- Increases code complexity at data handling boundaries
- Legacy data formats may need modification to accommodate checksum fields

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A scientific research institution discovered that its legacy data archival system had been silently corrupting a small percentage of files during storage, undetectable by the single CRC32 checksum it used. By adding a second SHA-256 checksum and verifying both on every read, the team identified 47 corrupted files in the archive that had passed the CRC32 check alone. For the ongoing data migration to a new storage platform, dual checksums provided confidence that every file was transferred with perfect fidelity.
