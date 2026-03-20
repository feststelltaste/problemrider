---
title: Redundant Data Storage
description: Storing data on multiple media or systems
category:
- Database
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/redundant-data-storage
problems:
- single-points-of-failure
- silent-data-corruption
- system-outages
- data-migration-integrity-issues
- unbounded-data-growth
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Implement database replication to maintain copies of critical data on separate storage systems
- Use RAID configurations for local storage to protect against individual disk failures
- Replicate data across geographic locations for disaster recovery scenarios
- Choose appropriate replication strategies: synchronous for zero data loss, asynchronous for performance
- Verify data consistency between replicas regularly using automated comparison tools
- Design the application to read from replicas to distribute load and provide automatic failover
- Document recovery procedures for switching to replica data when primary storage fails

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Protects against data loss from hardware failures, corruption, or disasters
- Enables continued operations when primary storage becomes unavailable
- Read replicas can improve query performance for legacy systems under heavy load
- Provides a foundation for disaster recovery and business continuity

**Costs and Risks:**
- Storage and infrastructure costs multiply with each additional copy
- Replication lag in asynchronous setups can cause stale reads or data conflicts
- Managing replication topology adds operational complexity
- Legacy applications may not handle read/write splitting without modification

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legal firm's legacy document management system stored all client documents on a single NAS device. When the device experienced a controller failure, the firm lost access to documents for two business days while awaiting replacement parts. After recovery, the team implemented redundant storage with real-time replication to a secondary NAS and nightly replication to cloud object storage. The next hardware failure triggered automatic failover to the secondary NAS within minutes, and the cloud copy provided an additional safety net for disaster scenarios.
