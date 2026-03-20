---
title: Virtualization
description: Isolate applications with their own OS instance to prevent resource and dependency conflicts
category:
- Operations
- Architecture
quality_tactics_url: https://qualitytactics.de/en/compatibility/virtualization
problems:
- deployment-environment-inconsistencies
- dependency-version-conflicts
- shared-dependencies
- configuration-drift
- poor-system-environment
- resource-contention
- technology-lock-in
layout: solution
---

## How to Apply ◆

- Migrate legacy applications from bare-metal shared servers to individual virtual machines, giving each application its own OS and dependency stack.
- Use infrastructure-as-code tools (Terraform, Ansible) to define and provision virtual environments reproducibly.
- Create VM images that capture the exact OS, runtime, and library versions a legacy application requires.
- Use snapshots for safe rollback when applying patches or configuration changes to legacy systems.
- Consolidate underutilized physical servers through virtualization to reduce hardware costs while maintaining isolation.
- Consider containerization (Docker) for lighter-weight isolation where the legacy application's OS requirements allow it.

## Tradeoffs ⇄

**Benefits:**
- Eliminates dependency conflicts between applications that require different library or runtime versions.
- Enables consistent environment reproduction across development, staging, and production.
- Provides isolation so that one application's resource consumption does not affect others.
- Simplifies disaster recovery through VM snapshots and image-based backups.

**Costs:**
- Adds overhead for managing virtualization infrastructure (hypervisor, image storage, networking).
- VMs consume more resources than containers due to full OS overhead per instance.
- Legacy applications with hardware-specific dependencies may not virtualize cleanly.
- Requires operational skills in virtualization platforms that teams may need to acquire.
- Licensing costs for operating systems and virtualization platforms can be significant.

## Examples

A government agency runs multiple legacy applications on shared Windows servers where conflicting .NET Framework versions and DLL dependencies cause frequent deployment failures. By virtualizing each application into its own VM with a fixed OS image, dependency conflicts are eliminated. The infrastructure team uses Ansible to provision VMs from versioned templates, ensuring that development environments match production exactly. When a critical legacy application needs an older runtime that conflicts with security patches required by another application, the isolation provided by virtualization allows both to coexist without compromise. The VM snapshot capability also gives the team confidence to attempt upgrades, knowing they can roll back within minutes.
