---
title: Customization Under Version Control
description: Export a packaged system's configuration and custom logic into text artifacts
  so they can be diffed, reviewed, reverted, and deployed like any other code.
category:
- Operations
- Process
- Code
problems:
- customization-outside-version-control
- low-code-customization-sprawl
- configuration-drift
- manual-deployment-processes
- configuration-chaos
- authorization-role-explosion
- lack-of-ownership-and-accountability
- invisible-nature-of-technical-debt
- regression-bugs
- slow-incident-resolution
- upgrade-blocked-by-customization
- inadequate-configuration-management
- core-modification-of-standard-software
- implementation-partner-dependency
layout: solution
related_solutions:
- slug: ci-cd-pipeline
  similarity: 0.7
- slug: infrastructure-as-code
  similarity: 0.7
- slug: explicit-extension-points
  similarity: 0.65
- slug: code-review-process-reform
  similarity: 0.65
- slug: self-service-developer-platform
  similarity: 0.65
- slug: development-workflow-automation
  similarity: 0.65
---

## Description

Customization under version control means extracting a packaged system's configuration and custom logic from its internal storage into text artifacts held in a repository, and treating those artifacts as the authoritative source from which environments are built. The point is not tidiness. Version control is the substrate on which review, reproducibility, traceability, and revert all depend, and none of them can be applied to state that exists only inside a running system. Teams that hold their own code to a high standard frequently operate their packaged systems with none of these practices, not from carelessness but because the platform never presented the option. Establishing the export is what makes every other practice available; until it exists, no amount of process discipline can compensate.

## How to Apply ◆

> The obstacle is almost never that the platform cannot export, but that nobody has ever made the export the source rather than a backup.

- **Find out what the platform can already export.** Most enterprise packages offer transport, migration, or serialization facilities intended for moving changes between environments. These usually produce something that can be committed, even when the format is unpleasant.
- **Establish the direction of authority explicitly.** The repository is the source and environments are built from it, or the system is the source and the repository is a record. The first is the goal; the second is a legitimate first step, and confusing the two produces a repository nobody trusts.
- **Start with what changes most and hurts most**, typically workflow definitions, scripts, and form logic. Attempting to bring the entire configuration surface under control at once produces a stalled project.
- **Make the exported form as readable as the platform allows.** Where the export is a binary or an opaque blob, invest in a conversion that produces something diffable — the diff is where most of the value is, and an uncomparable artifact delivers little.
- **Introduce review before deployment**, not before the change is made in a development environment. Requiring review before anyone may experiment removes the platform's main advantage; requiring it before anything reaches production restores the control that was missing.
- **Automate promotion between environments** from the repository. Manual promotion keeps the repository advisory, and an advisory repository drifts within weeks.
- **Detect drift continuously** by comparing the running configuration against the repository on a schedule. Direct production changes will happen; the question is whether they are noticed.
- **Restrict who may change production directly**, and log it when they do. Emergency access is legitimate and should leave a record that prompts the change to be brought back into the repository.
- **Prove it with a rebuild.** Reconstructing a working environment from the repository alone is the only real test of whether the source is authoritative, and it usually fails informatively the first time.

## Tradeoffs ⇄

> Bringing packaged customization under version control restores the practices the platform removed, at the cost of building an export and deployment path the vendor did not provide.

**Benefits:**

- Changes become reviewable before they reach production, which is the single largest quality improvement available in these environments.
- The customization inventory becomes listable and searchable, so it can be counted, assessed, and reduced.
- Reverting a change becomes possible, rather than a reconstruction from memory.
- Environments can be rebuilt from a known state, which changes both disaster recovery and the ability to test an upgrade realistically.
- Authorship and history make it possible to ask why something is configured as it is, and to have an addressee for the answer.

**Costs and Risks:**

- The export and promotion path is genuine engineering work that the vendor does not support and may break with a release.
- Some platform state cannot be exported meaningfully at all, so coverage will be partial and the boundary must be documented or the repository will be trusted further than it deserves.
- Introducing review adds latency to changes that administrators are used to making instantly, and this is felt as a loss.
- A repository that drifts from reality is worse than none, because decisions are made against it; keeping it authoritative requires the drift detection to actually be acted upon.
- Exported formats are frequently verbose and machine-oriented, so diffs may be large and hard to read even once they exist.

## How It Could Be

An IT service management platform had all its workflow and scripted logic in the platform database, maintained by four administrators with no review step. The team built an export of scripts and workflow definitions into a repository, initially as a nightly snapshot with no authority — purely a record. That alone changed things within a month: the snapshot diff became the answer to "what changed," which had previously required asking people, and it immediately surfaced 310 fragments referencing fields that no longer existed. Six months later the direction was reversed, with promotion to production driven from the repository and direct production changes restricted to a break-glass path that logged and alerted.

The rebuild test was where the value became undeniable. Their disaster recovery plan had assumed a replacement instance could be configured from documentation. Attempting it from the repository took two days and failed on three categories of state the export did not cover — integration credentials, scheduled job definitions, and a set of platform-level settings. All three were then handled explicitly, two by extending the export and one by documenting it as a manual step with a checklist. The organization's actual recovery position improved substantially, and the finding came from an exercise that the previous arrangement had made impossible to even attempt.
