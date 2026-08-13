---
title: Fit-to-Standard Principle
description: Make adopting the product's own process the default, and require every
  deviation to be justified, sized, and approved by someone who carries its cost.
category:
- Business
- Process
- Architecture
problems:
- process-software-misfit
- reimplemented-standard-functionality
- excessive-customization
- core-modification-of-standard-software
- upgrade-blocked-by-customization
- inefficient-processes
- inadequate-requirements-gathering
- eager-to-please-stakeholders
- increased-cost-of-development
- high-maintenance-costs
- gold-plating
- feature-creep
- voided-vendor-support
layout: solution
related_solutions:
- slug: explicit-extension-points
  similarity: 0.7
- slug: variant-consolidation
  similarity: 0.7
- slug: customization-cost-attribution
  similarity: 0.65
- slug: standard-software
  similarity: 0.65
- slug: change-management-process
  similarity: 0.65
- slug: evolutionary-requirements-development
  similarity: 0.6
---

## Description

The fit-to-standard principle inverts the default in a packaged software implementation: the product's process is adopted unless there is a stated reason not to, rather than the product being adapted unless there is a reason not to. The inversion matters because defaults determine outcomes when nobody is watching. Under the usual arrangement, a requirement expressed as current practice becomes a customization by inertia, and the cumulative result is a heavily adapted system that reproduces the past. Under fit-to-standard, the same requirement has to survive a question — what does the product do here, and why is that insufficient — and a substantial share of requirements do not survive it, because they were descriptions of habit rather than statements of need.

## How to Apply ◆

> Most requirements in a package implementation describe how the organization works today, and the value of the principle is that it forces someone to ask whether that is worth preserving.

- **Establish what the standard does before gathering requirements**, not after. A workshop that begins by demonstrating the product's process produces a different set of requirements than one that begins by documenting the current one.
- **Require a stated reason for every deviation**, in a fixed form: what the standard does, why it does not suffice, what it would cost the business to adapt instead, and who accepts the ongoing cost of the difference.
- **Distinguish genuinely differentiating processes from merely habitual ones.** An organization's competitive advantage is worth customizing for; its accounts payable approval sequence almost never is. Most deviation requests concern the second category.
- **Put the approval with someone who carries the consequence.** A deviation approved by the requesting department costs them nothing; approved by whoever owns the upgrade budget, it is a real decision.
- **Attach the lifetime cost to the request**, not just the build estimate. Implementation is a fraction of the total, and presenting only that fraction guarantees systematic underestimation of every customization ever proposed.
- **Timebox the challenge.** Fit-to-standard becomes obstruction if every request triggers an extended investigation. A defined short evaluation, with a default answer if it is not completed, keeps it workable.
- **Record the deviations that are approved** as a maintained register with reasons, so the accumulated set can be reviewed later and so future upgrades know what they are carrying.
- **Revisit deviations at each major release.** The vendor may have closed the gap, in which case the deviation can be retired — and nobody will notice unless someone checks.
- **Give the principle a named owner with authority.** Without someone empowered to say no, the default reverts to accommodation within the first few contested requests.

## Tradeoffs ⇄

> Defaulting to the standard preserves upgradeability and removes a large amount of unnecessary adaptation, but it requires the organization to change how it works and someone with authority to insist.

**Benefits:**

- Customization volume falls substantially, and with it the cost of every future upgrade and change.
- Requirements are examined rather than transcribed, which frequently reveals that a described need was a description of habit.
- The organization receives the vendor's process improvements automatically instead of maintaining a fork of an older way of working.
- Deviations that are approved are documented with reasons, which makes the accumulated set reviewable rather than archaeological.
- Implementation is faster, because configuring the standard is quicker than building an alternative to it.

**Costs and Risks:**

- Business processes have to change, which is disruptive, is resisted, and requires authority that software projects frequently do not have.
- Applied dogmatically, it forces genuinely differentiating processes into a generic mould and can damage a real competitive advantage.
- The evaluation adds latency to every requirement, and if the process is heavy it becomes an obstacle that people route around.
- The product's standard process is not always good; vendors encode assumptions that may not suit your sector or scale.
- Change management for the affected staff is a substantial cost that is routinely omitted from the comparison, making fit-to-standard look cheaper than it is.

## How It Could Be

An organization implementing a document and records platform ran requirements workshops that began with a demonstration of the standard process, followed by the question of where it would not work. Of 140 requirements raised, 96 were met by configuration once participants had seen what the product did. Of the remaining 44, the deviation form — what the standard does, why it is insufficient, what changing the process would cost — eliminated a further 19, in most cases because the requesting department could not answer the second question with anything beyond current practice. Twenty-five deviations were approved, each recorded with a reason and an accepting owner. A comparable implementation at a sister organization two years earlier, run the conventional way, had produced 130 customizations.

The revisit discipline paid off later and unexpectedly. At the second major release after go-live, the deviation register was reviewed against the vendor's release notes. Four of the 25 deviations had been closed by standard functionality the vendor had since shipped, and one had become unnecessary because the department it served had reorganized. Retiring those five took three weeks. Without the register, none of them would have been noticed — the deviations would simply have been carried forward as part of the system, as the four hundred at their sister organization were.
