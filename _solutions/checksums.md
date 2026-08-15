---
title: Checksums
description: Checksum calculation for detecting data errors or changes
category:
- Security
- Code
problems:
- silent-data-corruption
- data-migration-integrity-issues
- cross-system-data-synchronization-problems
- insecure-data-transmission
- dma-coherency-issues
- retention-obligations-block-change
layout: solution
related_solutions:
- slug: redundant-checksums
  similarity: 0.9
- slug: error-correction-codes
  similarity: 0.8
- slug: data-integrity
  similarity: 0.8
- slug: continuous-data-verification
  similarity: 0.8
- slug: fault-tolerant-data-structures
  similarity: 0.75
- slug: plausibility-checks
  similarity: 0.7
---

## Description

A checksum is a small, fixed-size value computed from a larger block of data using an algorithm such as CRC32 or SHA-256, such that any change to the original data is very likely to change the computed value, making checksums an efficient way to detect corruption or unauthorized modification without comparing the full data itself. They are generated at the point data is produced or transmitted and recomputed at the point it is consumed, with a mismatch signaling that the data was altered somewhere along the way. This is especially relevant in legacy modernization work involving data migration, where records move between systems with different encodings, precision handling, or storage formats, and silent corruption during transfer can go completely unnoticed until it surfaces much later as an inexplicable business discrepancy. Adding checksum verification at both ends of a migration batch converts that risk from an invisible, cumulative problem into an immediately detectable one, since a failed checksum comparison flags exactly which records need investigation before they are trusted downstream. The same mechanism protects ongoing data flows in legacy systems — file transfers, message queues, API payloads — where transmission errors over unreliable channels might otherwise corrupt data without any visible symptom. Checksums are not a substitute for stronger integrity guarantees like cryptographic signing when tampering rather than accidental corruption is the concern, and their protective value is limited to whatever algorithm strength is chosen for the task.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify critical data flows where corruption or tampering could occur: file transfers, database migrations, API communication, and message queues
- Choose appropriate checksum algorithms based on requirements (CRC32 for error detection, SHA-256 for integrity verification)
- Add checksum generation at data source points and verification at consumption points
- Include checksums in data migration scripts to verify that source and target data match after migration
- Implement checksum validation in file upload/download processes to detect transmission corruption
- Store checksums alongside data records for periodic integrity audits
- Log checksum mismatches with sufficient context for investigation and remediation

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Detects data corruption early before it propagates through the system
- Provides confidence in data integrity during migrations between legacy and modern systems
- Enables verification of data transmission completeness across unreliable networks
- Creates an audit trail for data changes and integrity verification

**Costs and Risks:**
- Adds computational overhead for checksum calculation on high-volume data paths
- Checksum storage requires additional space alongside the actual data
- False sense of security if using weak checksum algorithms that do not detect certain error patterns
- Retrofitting checksum verification into existing data flows requires modifying both sender and receiver

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

During a legacy database migration from an on-premises SQL Server to a cloud-hosted PostgreSQL instance, a financial services team discovered that 0.3% of records had been silently corrupted during transfer due to character encoding mismatches. After fixing the encoding issue, they added SHA-256 checksums to every batch of migrated records. The migration script computed checksums on the source, transferred the data, recomputed checksums on the target, and compared them before committing each batch. This approach caught two additional corruption patterns related to decimal precision differences and ensured that all 12 million records arrived intact.
