---
title: Customization Outside Version Control
description: Configuration and custom logic live inside the package's own database, so they cannot be diffed, reviewed, reproduced, or traced to who changed what.
category:
- Operations
- Process
- Code
related_problems:
solutions:
- customization-under-version-control
- version-control
- ci-cd-pipeline
- infrastructure-as-code
- code-reviews
- environment-parity
- configuration-checks
- immutable-infrastructure
- clear-ownership-model
- audit-trail-management
- role-model-rationalization
layout: problem
---

## Description

In many packaged systems the customizations — configuration, custom fields, workflow definitions, scripts, report layouts, role assignments — are stored inside the product's own database rather than as files. There is no repository, no commit, no diff, and often no record of who changed what or why. The consequence is that every engineering practice built on version control simply does not apply: changes cannot be reviewed before they take effect, an environment cannot be reproduced from a known state, a change cannot be reverted except by remembering what it was, and the total customization inventory cannot be listed. Teams that maintain rigorous discipline in their own codebases frequently operate their packaged systems with none of it, without noticing the inconsistency, because the tooling never offered the option.

## Indicators ⟡

- Nobody can answer what changed in the system last month without asking people
- Test and production configurations differ in ways that are discovered rather than known
- A change is made directly in production because that is where the configuration lives and there is no other way to apply it
- Reverting a change means somebody reconstructing the previous value from memory or a screenshot
- There is no review step before a configuration change takes effect, and no record that one occurred
- Recreating a working environment from scratch is considered impossible, or takes weeks of manual comparison
- The count of custom objects, fields, or workflows is unknown and can only be established by exporting and counting

## Symptoms ▲

- [Configuration Drift](configuration-drift.md)
<br/>  Without a single authoritative source, environments diverge continuously and the divergence is only found when something behaves differently.
- [Manual Deployment Processes](manual-deployment-processes.md)
<br/>  Moving a change between environments means repeating it by hand, which is slow, error-prone, and unverifiable.
- [Invisible Nature of Technical Debt](invisible-nature-of-technical-debt.md)
<br/>  Customization that cannot be listed cannot be assessed, so its accumulated weight remains invisible to everyone including the people carrying it.
- [Regression Bugs](regression-bugs.md)
<br/>  Changes take effect without review and without a revert path, so a mistaken adjustment reaches users and stays until someone reconstructs the prior state.
- [Lack of Ownership and Accountability](lack-of-ownership-and-accountability.md)
<br/>  Without an authorship record, no change has an owner, and questions about why something is configured a certain way have no addressee.
- [Poor Documentation](poor-documentation.md)
<br/>  The configuration is its own documentation and it is not readable, so understanding the system requires clicking through screens rather than reading.
- [Knowledge Silos](knowledge-silos.md)
<br/>  What the system does is known only by those who made the changes, because the changes left no artifact anyone else can read.
- [Slow Incident Resolution](slow-incident-resolution.md)
<br/>  Diagnosis cannot begin from what changed recently, since that question has no answer, so investigations start from scratch every time.

## Causes ▼

- [Vendor Lock-In](vendor-lock-in.md)
<br/>  The package stores its configuration internally by design, and exporting it into a reviewable form requires effort the vendor does not support.
- [Legacy Skill Shortage](legacy-skill-shortage.md)
<br/>  Staff administering the package often come from an operations rather than a development background, and version control is not part of the practice they learned.
- [Short-Term Focus](short-term-focus.md)
<br/>  Establishing an export and deployment pipeline delivers nothing visible, while making the change directly delivers it today.

## Detection Methods ○

- Ask for a diff of the configuration between two environments; the difficulty of producing one measures the problem directly
- Attempt to answer what changed in the last thirty days and how long the answer takes
- Check whether any change to the package's configuration passes through a review before taking effect
- Try to rebuild a test environment from a defined source and record what has to be done by hand
- Look for whether the package offers an export format, and whether anyone is using it
- Count how many people can change production configuration directly and whether their changes are logged

## Examples

An IT service management platform had been in use for six years, configured by four administrators across three teams. Workflows, form logic, approval rules, and several hundred scripted behaviors lived entirely in the platform's database. When a change to an approval routing began sending requests to a department that no longer existed, the investigation could not establish when the routing had been set, by whom, or what it had been before. Three administrators each remembered a different original value. The resolution took eleven days, most of which was spent reconstructing intent rather than making the change. The organization's application code, by contrast, was under review, tested, and deployed through a pipeline — the same engineers simply had no equivalent for the platform that ran their incident process.

The reproducibility cost surfaced during a disaster recovery exercise. The plan assumed a replacement instance could be built and configured from documentation. The exercise established that the documentation described roughly a third of the live configuration, that the remainder existed only in the running system, and that the organization's actual recovery position depended entirely on the database backup being restorable. Nobody had considered this a customization problem; it had been filed as an infrastructure concern for four years.
