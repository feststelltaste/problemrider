---
title: Core Modification of Standard Software
description: The vendor's own code has been altered directly instead of extended through supported mechanisms, so every update collides with local changes.
category:
- Architecture
- Dependencies
- Code
related_problems:
solutions:
- explicit-extension-points
- fit-to-standard-principle
- customization-under-version-control
- large-scale-refactoring
- characterization-tests
- change-impact-analysis
- technical-debt-assessment
- debt-remediation-estimation
- customization-cost-attribution
- variant-consolidation
- vendor-management-practice
- modernization-options-comparison
- cost-of-delay
layout: problem
---

## Description

Core modification occurs when a purchased software package is adapted by editing the vendor's own delivered code rather than by using the extension mechanisms the vendor provides. It is usually the fastest route at the moment it is chosen: the required behavior sits inside a delivered routine, changing it there takes an hour, and building the same thing through a supported extension point takes a week. The cost arrives later and permanently. Every subsequent update from the vendor overwrites or conflicts with the modification, so each upgrade becomes an exercise in reconciling two sets of changes to the same code. Because the organization now maintains a fork of software it did not write and does not fully understand, the fork can never be reconciled — only carried.

## Indicators ⟡

- Applying a vendor update requires a merge, and someone has to decide per conflict which version wins
- There is a list, formal or informal, of "objects we have changed" that must be consulted before any upgrade
- Upgrade projects are scheduled in months and involve external consultants regardless of how minor the release is
- Vendor documentation does not describe how your system behaves, and staff have learned to distrust it
- Modifications carry comments from developers who left years ago, explaining a business rule that no longer applies
- The organization runs a version well behind the current release and has no dated plan to close the gap

## Symptoms ▲

- [Obsolete Technologies](obsolete-technologies.md)
<br/>  Because each upgrade must reconcile the fork, upgrades are deferred and the installed version falls progressively further behind what the vendor supports.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  The organization maintains code it did not write, in a system it does not fully understand, in addition to its own extensions.
- [Testing Complexity](testing-complexity.md)
<br/>  Modified vendor code cannot be assumed to behave as documented, so verification must cover standard behavior the vendor has already tested.
- [Regression Bugs](regression-bugs.md)
<br/>  Reconciling a vendor update with local modifications reintroduces defects the vendor fixed, or removes local behavior that something depended on.
- [Fear of Breaking Changes](fear-of-breaking-changes.md)
<br/>  Nobody is confident about what a modified routine is now responsible for, so changes near it are avoided even when they are needed.
- [Long Release Cycles](long-release-cycles.md)
<br/>  The reconciliation and regression effort attached to every vendor release turns routine updates into projects that can only be undertaken rarely.
- [Vendor Dependency Entrapment](vendor-dependency-entrapment.md)
<br/>  A heavily modified installation cannot be replaced by a comparable product without redoing every modification, which removes the option of leaving.
- [Difficult to Understand Code](difficult-to-understand-code.md)
<br/>  Vendor code carrying local edits is legible neither as the vendor's product nor as an in-house system, and no documentation describes the combination.

## Causes ▼

- [Excessive Customization](excessive-customization.md)
<br/>  When the volume of required adaptation exceeds what the extension mechanisms comfortably support, editing the core becomes the path of least resistance.
- [Market Pressure](market-pressure.md)
<br/>  A commitment made during procurement or a deal has to be met by a date, and the supported route is slower than the date allows.
- [Eager to Please Stakeholders](eager-to-please-stakeholders.md)
<br/>  Requests are accepted without evaluating whether they can be met within the standard, and the technical consequence is discovered afterwards.
- [Inadequate Requirements Gathering](inadequate-requirements-gathering.md)
<br/>  Nobody established whether the requirement could be met by configuration, so development began where the code happened to be.
- [Legacy Skill Shortage](legacy-skill-shortage.md)
<br/>  Staff who know the package's extension framework are scarce, while anyone can edit a routine, so the unsupported route is also the available one.
- [Project Authority Vacuum](project-authority-vacuum.md)
<br/>  No one has the standing to refuse a modification or to insist on the supported mechanism, so the decision is made by whoever is implementing.

## Detection Methods ○

- Use the package's own facilities to list objects that differ from the delivered state; most enterprise packages can report this directly
- Compare the installed codebase against a clean installation of the same version and count the differing objects
- Review the last upgrade's effort breakdown and identify how much went to reconciling modifications rather than to testing or training
- Check whether support requests have been declined or qualified on the grounds that the system is modified
- Count how many modified objects have no documented reason, no owner, and no test
- Ask whether anyone can produce, within a day, a list of every modification and why it exists

## Examples

A manufacturer running an enterprise resource planning package had modified 340 delivered objects over fourteen years. Most modifications were small — an extra field on a screen, an additional validation, a changed sort order — and each had been the sensible choice at the time. The cumulative effect was that a vendor release which the vendor described as a routine update took their team five months, of which roughly four were spent reconciling modifications and regression testing the result. They were four major versions behind. Two of the modifications, when investigated, implemented behavior the standard product had gained in a release six years earlier, so the fork was being maintained to preserve a worse version of a feature the vendor now shipped.

A different pattern appeared in a document management deployment. The organization had modified the delivered retention routine to accommodate a rule specific to one department. Years later, a regulatory change required an adjustment to retention handling, which the vendor delivered as a patch. Applying it would have removed the local rule; not applying it left the organization out of compliance. Neither option was available without a project, and the department whose rule had prompted the original modification had been reorganized out of existence four years earlier — a fact nobody established until the reconciliation forced someone to ask who still needed the behavior.
