---
title: Backup and Recovery
description: Ensure regular backup and recoverability of data
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/backup-and-recovery
problems:
- system-outages
- silent-data-corruption
- data-migration-integrity-issues
- missing-rollback-strategy
- deployment-risk
- regulatory-compliance-drift
- configuration-drift
layout: solution
---

## How to Apply ◆

> Legacy systems often have backup procedures that were configured years ago and never tested for actual recoverability. Backup and recovery ensures that data can be reliably restored after failures, corruption, or security incidents.

- Inventory all data stores in the legacy system — databases, file systems, configuration files, application state, certificates, and encryption keys — and verify that each has an appropriate backup strategy.
- Implement the 3-2-1 backup rule: maintain at least three copies of critical data, on at least two different media types, with at least one copy stored off-site or in a different availability zone.
- Define Recovery Point Objectives (RPO) and Recovery Time Objectives (RTO) for each data store based on business requirements. The RPO defines the maximum acceptable data loss (how often backups run), and the RTO defines the maximum acceptable downtime (how quickly restores must complete).
- Test backup restoration regularly — at minimum quarterly — by performing actual restore operations to a separate environment and verifying data integrity. An untested backup is not a backup; it is a hope.
- Implement automated backup verification that checks backup file integrity (checksums, size validation) after each backup completes. Many legacy backup failures are only discovered when a restore is attempted.
- Secure backup storage with encryption at rest, access controls, and immutability protection to prevent ransomware from encrypting or deleting backup copies.
- Document the recovery procedure step-by-step, including the order of operations for restoring multiple interdependent systems. Ensure that multiple team members can execute the procedure, not just the person who set up the backups.

## Tradeoffs ⇄

> Reliable backup and recovery provides the ultimate safety net against data loss, but it requires ongoing investment in storage, testing, and process maintenance.

**Benefits:**

- Provides recoverability from hardware failures, software bugs, human errors, ransomware attacks, and natural disasters.
- Supports compliance requirements that mandate data recoverability and business continuity planning.
- Enables confident deployment and migration operations, knowing that data can be restored if changes cause corruption.
- Reduces the business impact of security incidents by enabling restoration to a known-good state.

**Costs and Risks:**

- Backup storage costs grow with data volume and retention period, particularly for legacy systems with large, growing databases.
- Backup windows for large legacy databases can impact system performance during backup execution.
- Backups that are not regularly tested may fail when needed due to corruption, incompatible formats, or incomplete capture of all required data.
- Recovery procedures for complex legacy systems with multiple interdependent data stores are error-prone and time-consuming without thorough documentation and practice.

## Examples

> The following scenarios illustrate how backup and recovery practices protect legacy systems from data loss.

A legacy ERP system's database becomes corrupted after a storage controller failure. The operations team attempts to restore from the most recent backup and discovers that the backup job has been silently failing for three weeks due to a full backup disk that no one monitored. The company loses three weeks of transaction data. After this incident, the team implements automated backup monitoring that verifies each backup completes successfully, checks the backup file size against expected ranges, and alerts immediately on any failure. They also institute monthly restore tests where the complete database is restored to a test environment and a set of validation queries confirms data integrity. When a subsequent disk failure occurs eight months later, the restore completes in 4 hours from a backup taken 6 hours prior, well within the defined RPO of 24 hours and RTO of 8 hours.

A legacy content management system is hit by a ransomware attack that encrypts the application database and all attached file storage. The existing backups are stored on a network share that the ransomware also encrypts. The organization loses all content created in the past 30 days. After recovery, the team implements immutable backup storage using write-once cloud storage, ensuring that backups cannot be modified or deleted for a minimum retention period of 90 days. They also add air-gapped backups that are physically disconnected from the network. A recovery drill performed the following quarter demonstrates that the full system can be restored from immutable backups within the 12-hour RTO, and the backups survive a simulated ransomware scenario in the test environment.
