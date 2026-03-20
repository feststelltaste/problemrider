---
title: Restore Points
description: Regularly back up the system state
category:
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/restore-points
problems:
- missing-rollback-strategy
- deployment-risk
- system-outages
- configuration-drift
- data-migration-integrity-issues
- fear-of-change
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Create system state snapshots before any significant change (deployment, migration, configuration update)
- Use database point-in-time recovery capabilities to enable restoration to any moment within a retention window
- Capture virtual machine or container snapshots as lightweight restore points for infrastructure-level rollback
- Store restore points with metadata describing what change prompted their creation
- Automate restore point creation as part of deployment pipelines so it is never skipped
- Test restoration from restore points periodically to verify they produce a working system
- Define retention policies that balance storage costs with the need for historical recovery options

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables rapid rollback when changes cause unexpected problems in legacy systems
- Reduces the risk of deployments and migrations by providing a known-good fallback state
- Builds confidence for making changes to legacy systems
- Provides a clear recovery path that reduces incident stress

**Costs and Risks:**
- Restore points consume storage that grows with system size and change frequency
- Restoring to a previous point may lose legitimate data or transactions created after the snapshot
- Point-in-time recovery may not capture all system state (external integrations, message queues)
- Teams may use restore points as a crutch instead of investing in proper testing

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A university's legacy student information system required a complex database schema migration to support new enrollment features. The team created a full database restore point and VM snapshot before starting the migration. When the migration script encountered an unforeseen constraint violation halfway through, corrupting referential integrity in several tables, the team restored to the pre-migration state within 20 minutes rather than spending hours attempting manual data repair. They fixed the migration script, tested it against a copy of the restored database, and ran it successfully on the second attempt.
