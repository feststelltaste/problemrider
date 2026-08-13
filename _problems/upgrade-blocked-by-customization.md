---
title: Upgrade Blocked by Customization
description: Vendor releases cannot be applied because the accumulated local adaptation
  would have to be reconciled and revalidated each time.
category:
- Dependencies
- Operations
- Process
related_problems:
- slug: core-modification-of-standard-software
  similarity: 0.7
- slug: excessive-customization
  similarity: 0.65
- slug: vendor-dependency-entrapment
  similarity: 0.65
- slug: voided-vendor-support
  similarity: 0.6
- slug: reimplemented-standard-functionality
  similarity: 0.55
- slug: schema-evolution-paralysis
  similarity: 0.55
solutions:
- fit-to-standard-principle
- explicit-extension-points
- continuous-dependency-updates
- characterization-tests
- automated-tests
- regression-testing
- parallel-run
- staged-investment-with-decision-gates
- cost-of-delay
- risk-quantification
- variant-consolidation
- no-regret-moves
- modernization-options-comparison
- customization-under-version-control
layout: problem
---

## Description

Upgrade blockage occurs when the effort to bring an installation onto a new vendor release exceeds what the organization is willing to spend, so releases are skipped. Each skipped release makes the next one harder, because the reconciliation now spans several versions of vendor change at once. The condition compounds until the installed version leaves vendor support, at which point security patches stop, the available skill pool shrinks, and the eventual upgrade is no longer an upgrade but a migration. What makes this distinct from ordinary deferral is that the blockage is self-inflicted and cumulative: the organization is not waiting for anything external, and every month of waiting increases the price of the decision it is avoiding.

## Indicators ⟡

- The installed version is more than one major release behind, and the gap has widened over time
- Upgrade effort is estimated in months, and the estimate has grown between successive attempts
- An upgrade has been planned and cancelled at least once
- Vendor support responses increasingly begin by asking you to reproduce the issue on a current version
- Nobody can state the total effort without a discovery exercise costing weeks
- New capability the business wants is available in a release the organization cannot reach

## Symptoms ▲

- [Obsolete Technologies](obsolete-technologies.md)
<br/>  The installation ages out of vendor support, and the runtime, database, and operating system it depends on age with it.
- [Regulatory Compliance Drift](regulatory-compliance-drift.md)
<br/>  Regulatory changes delivered by the vendor as product updates cannot be received, so compliance has to be built locally or is simply missing.
- [Legacy Skill Shortage](legacy-skill-shortage.md)
<br/>  The pool of people who know an unsupported version shrinks continuously, and new hires have no reason to have learned it.
- [Delayed Value Delivery](delayed-value-delivery.md)
<br/>  Capability the vendor has already built and the organization has already paid for remains unavailable to users indefinitely.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Defects the vendor has fixed must be worked around locally, and extended support contracts for old versions carry a premium.
- [Modernization ROI Justification Failure](modernization-roi-justification-failure.md)
<br/>  The upgrade grows into a figure large enough that no business case succeeds, which guarantees further deferral.
- [Vendor Dependency Entrapment](vendor-dependency-entrapment.md)
<br/>  An installation stranded on an unsupported version has neither a viable upgrade path nor a viable replacement path.

## Causes ▼

- [Core Modification of Standard Software](core-modification-of-standard-software.md)
<br/>  Modified vendor objects must be reconciled against every release, which is the single largest component of most upgrade estimates.
- [Excessive Customization](excessive-customization.md)
<br/>  The volume of local adaptation determines how much must be revalidated, and it grows continuously while the upgrade is deferred.
- [Customization Outside Version Control](customization-outside-version-control.md)
<br/>  Where the customization inventory cannot be listed, the upgrade cannot be scoped, so estimates are large and defensive.
- [Increased Manual Testing Effort](increased-manual-testing-effort.md)
<br/>  Without automated regression coverage, revalidating a full product installation is a manual exercise measured in person-months.
- [Short-Term Focus](short-term-focus.md)
<br/>  In any given quarter the upgrade is less urgent than whatever is being delivered, and this comparison is made repeatedly with the same result.

## Detection Methods ○

- Record the installed version, the current release, and the date support ends for what you run
- Plot the gap between installed and current version over the last five years; the trend matters more than the level
- Compare the effort estimates of successive upgrade attempts to establish whether the reconciliation burden is growing
- Measure what share of the last upgrade went to reconciling modifications versus testing versus training
- Count how many vendor-delivered fixes have been reimplemented locally because the release carrying them could not be applied
- Ask what would have to be true for an upgrade to be routine, and treat the gap between that and reality as the actual backlog

## Examples

An enterprise resource planning installation had last been upgraded six years earlier. Two subsequent attempts had been planned and cancelled: the first when discovery established the effort at eight months against a four-month budget, the second when the consultancy's estimate came back higher than the first. By the third attempt the installed version had left mainstream support, the organization was paying an extended support premium, and a regulatory change to invoicing had to be implemented locally because it arrived as a vendor update they could not apply. The eventual programme took fourteen months. Roughly two thirds of the effort went to 340 modified vendor objects and to regression testing that had no automated foundation.

The compounding was visible in the numbers. The first cancelled attempt had estimated eight months. Four years and roughly 60 further customizations later, the same scope was estimated at thirteen. Nothing external had changed. The organization had spent four years making the decision it was avoiding more expensive, and at no point had anyone computed what the deferral was costing per month — which, when finally calculated during the third attempt, exceeded the annual cost of the extended support contract that had been treated as the price of waiting.
