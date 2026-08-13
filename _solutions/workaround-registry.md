---
title: Workaround Registry
description: "Record every workaround at the moment it is introduced \u2014 what it\
  \ hides, what it costs, and what would remove it \u2014 so that temporary fixes\
  \ stop becoming permanent invisibly."
category:
- Code
- Process
- Operations
problems:
- accumulation-of-workarounds
- workaround-culture
- increased-technical-shortcuts
- invisible-nature-of-technical-debt
- partial-bug-fixes
- increased-manual-work
- quality-compromises
- hidden-dependencies
- constant-firefighting
- delayed-bug-fixes
- high-technical-debt
- operational-overhead
layout: solution
related_solutions:
- slug: blameless-postmortems
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.7
- slug: technical-debt-backlog
  similarity: 0.7
- slug: knowledge-sharing-practices
  similarity: 0.7
- slug: code-review-process-reform
  similarity: 0.7
- slug: knowledge-base
  similarity: 0.65
---

## Description

A workaround registry is a single, maintained list of the compensating measures a system depends on: the manual step someone performs each month, the retry that masks an unreliable interface, the special case for one customer, the scheduled job that repairs data another job corrupts. Each entry records what problem it compensates for, what it costs to maintain, what would be required to remove it, and who introduced it and when. Workarounds are individually rational — under time pressure, compensating is usually correct — and collectively corrosive, because each one is invisible from outside and each one constrains what can be changed later. The registry does not discourage workarounds. It makes them countable, so that the organization can see the accumulated weight of decisions that each seemed sensible in isolation.

## How to Apply ◆

> The most expensive workarounds in a legacy system are usually the ones outside the code: the manual reconciliation, the spreadsheet, the daily check someone has performed for so long that nobody remembers it is a workaround.

- **Record the workaround at the moment it is created**, as part of the change that introduces it. Retrospective archaeology finds a fraction of them, and the person who introduced it is the only one who knows what it compensates for.
- Capture five fields and no more: **what it does, what problem it hides, what it costs to maintain, what would remove it, and when it was introduced.** More fields mean fewer entries, and coverage matters more than detail.
- Include **operational and organizational workarounds**, not just code. The manual step, the spreadsheet, the recurring calendar reminder, and the runbook entry that says "if this happens, do that" all belong. These are frequently the largest costs and the least visible.
- **Mark the workaround in the code itself** with a consistent, greppable marker linking to the registry entry. A developer encountering the code must be able to find out why it is there, and a comment saying "temporary" with no date or reference is worse than nothing.
- **Review the registry on a fixed cadence** — quarterly is usually right. The review asks two questions per entry: is the compensated problem still real, and has the cost changed. Workarounds regularly outlive their cause entirely, and finding those is the cheapest possible cleanup.
- **Feed removal candidates into the improvement budget.** The registry is only worth maintaining if entries occasionally leave it, and the ones to prioritize are those with high maintenance cost and low removal cost.
- **Report the count and the trend** alongside other health measures. A registry growing steadily is a system accumulating constraints, and the trend is a more persuasive figure to management than any individual entry.
- **Do not use it to assign blame.** A registry used to identify who cut corners will stop being filled in within a month, and the workarounds will continue while the record of them disappears.
- **Set expiry dates** where the workaround is genuinely meant to be temporary, and let the expiry trigger a review rather than an automatic removal. A stated date that passes is at least a visible decision to keep it.

## Tradeoffs ⇄

> The registry converts invisible accumulated constraint into a countable list, at the cost of discipline in maintaining it and the discomfort of an explicit record of compromise.

**Benefits:**

- Workarounds stop being invisible, which is the property that lets them accumulate indefinitely without any decision ever being made about them.
- Developers encountering unexplained code can find out why it exists, which prevents the common failure of removing a workaround and reintroducing the problem it hid.
- Obsolete workarounds get found. A substantial share compensate for problems that were fixed or systems that were retired years ago.
- The trend gives an early signal of degradation, often earlier than defect rates or cycle times.
- The genuine cost of deferred fixes becomes visible, which is one of the few effective arguments for addressing an underlying problem rather than compensating for it again.

**Costs and Risks:**

- Registries decay. One that is not reviewed becomes a stale list that nobody trusts, which is worse than none because it creates false confidence that workarounds are tracked.
- Recording is easy to skip precisely when it matters most — under the time pressure that produced the workaround in the first place.
- An explicit list of compromises can be used against the team by an unsympathetic reader, so it needs to be framed as an engineering instrument rather than a confession.
- Identifying non-code workarounds requires cooperation from business departments, who may not perceive their manual step as a workaround at all.
- Making workarounds visible without providing capacity to remove them produces frustration and a growing list that demonstrates only that nothing is being done.

## How It Could Be

A team maintaining a hospital billing system started recording workarounds after an incident in which a developer removed what looked like redundant validation and broke a data feed to an external laboratory system. Their first quarterly review covered 34 recorded entries plus 19 reconstructed from memory. Eleven compensated for problems that no longer existed: four referenced a payment provider replaced two years earlier, three worked around a database version that had since been upgraded, and one had been introduced for a customer who was no longer a customer. Removing those eleven took nine days and eliminated two recurring monthly manual steps. The most expensive single entry turned out to be organizational — a finance clerk spending roughly two days a month reconciling records because two subsystems disagreed about how to handle mid-month plan changes, a workaround that had been in place for six years and had never appeared in any technology discussion.

The trend measure changed how the team's degradation was discussed. Over eighteen months the registry grew from 34 to 61 entries, with the growth concentrated in one subsystem and accelerating after a period of deadline pressure. Presented to their director as a chart — workarounds added per quarter, alongside the count removed — it made an argument that the team's previous descriptions of accumulating technical debt had not. The response was not a modernization programme but something more useful: a standing rule that any workaround introduced under deadline pressure carried a mandatory removal review at the next quarterly, which halved the net growth rate over the following year.
