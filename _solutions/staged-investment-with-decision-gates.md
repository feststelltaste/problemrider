---
title: Staged Investment With Decision Gates
description: Fund modernization in tranches that each buy information, with a stated decision at every gate — including the decision to stop.
category:
- Management
- Business
- Process
problems:
- modernization-roi-justification-failure
- difficulty-quantifying-benefits
- modernization-strategy-paralysis
- history-of-failed-changes
- budget-overruns
- analysis-paralysis
- decision-paralysis
- incomplete-projects
- second-system-effect
- planning-credibility-issues
- system-stagnation
- inability-to-innovate
- poor-planning
- premature-technology-introduction
- technology-lock-in
- upgrade-blocked-by-customization
layout: solution
---

## Description

Staged investment funds a modernization in a sequence of small tranches, each of which is expected to reduce uncertainty rather than to deliver the whole outcome, and each of which ends at a gate where a stated decision is made: continue, change approach, or stop. It addresses the structural problem with large legacy proposals — the organization is asked to commit a large, uncertain sum against a benefit it cannot verify, on the strength of an estimate produced when least is known. That request is rationally declined, which is why so many well-founded modernization cases fail. Staging changes what is being asked for. The first tranche does not ask for approval of the modernization; it asks for a small amount to find out what the modernization would actually cost, with an explicit right to abandon it afterwards.

## How to Apply ◆

> A legacy modernization estimate produced before anyone has attempted a piece of it is not an estimate, and everyone in the room knows it — staging is the honest response to that.

- **Make the first tranche buy information, not progress.** Extract one small capability, migrate one table group, run one parallel comparison. The deliverable is a defensible estimate of the whole, produced from having done a representative piece.
- **Size each tranche to what the organization can afford to lose.** The test is whether the sponsor would be comfortable writing it off entirely. If not, it is too large and the gate will not function, because stopping will be politically impossible.
- **Define the gate criteria before the tranche starts**, in writing, including what result would mean stop. A gate whose only possible outcome is continue is a milestone, and the organization will treat it as one.
- **Make stopping a respectable outcome.** The first time a gate is used to stop something is what establishes whether the mechanism is real. If stopping is treated as failure, subsequent tranches will report success regardless of what happened.
- **Re-estimate the remainder at every gate**, using what the completed tranches actually cost rather than the original plan. Legacy estimates improve dramatically once a representative piece has been done, and the revised figure is the most valuable output of the early stages.
- **Sequence so that each tranche leaves something of value standing.** A stage that is only meaningful if the following stages happen recreates the all-or-nothing commitment inside the staged structure.
- **Report at gates in the sponsor's terms** — revised cost, revised benefit, what was learned, what the decision is — rather than as technical progress. A gate report that requires architectural knowledge to interpret will be approved without being understood, which defeats the mechanism.
- **Keep the gates infrequent enough to be meaningful.** Monthly gates become status meetings; quarterly ones force an actual decision. The right interval is roughly the time needed for the next tranche to change what is known.
- **Record the abandoned options.** A tranche that establishes that an approach will not work has produced a real result, and documenting it prevents the same approach being proposed again in two years by someone who was not there.

## Tradeoffs ⇄

> Staging makes large modernizations fundable by bounding the commitment, at the cost of overhead at each gate and the risk of an effort being stopped halfway for reasons unrelated to its merits.

**Benefits:**

- The initial ask is small enough to approve, which is frequently the difference between a modernization starting and being declined for a fourth year.
- Estimates improve rapidly, because each tranche produces evidence rather than analysis — and legacy cost estimates made without evidence are close to worthless.
- The organization's exposure is bounded at every point, which is what makes sponsors willing to back work with genuinely uncertain outcomes.
- Approaches that will not work are discovered early and cheaply, rather than at the point where too much has been spent to change course.
- Each gate produces a fresh, defensible business case, so support does not have to rest indefinitely on the credibility of the original one.

**Costs and Risks:**

- Gate preparation and review consume real effort and calendar time, and the overhead is disproportionate if the tranches are too small.
- Staged funding can be withdrawn at any gate for reasons that have nothing to do with the work — a budget round, a change of sponsor — leaving the system half-migrated, which is worse than either end state.
- Sequencing so that every stage stands alone is genuinely harder than a straight-through plan, and sometimes it is not possible.
- Organizations that treat every gate as an approval formality get the overhead without the decision, which is the most common failure mode.
- The staged approach can be slower and more expensive in total than a committed programme, when the programme would have succeeded.

## How It Could Be

A retailer's order management platform had been the subject of three modernization proposals over five years, each asking for between €4 and €7 million, each declined on the grounds that the estimate was not credible. The fourth proposal asked for €280,000 and four months, to extract one capability — product availability lookup — behind an interface, run it in parallel with the existing implementation, and report three things at the gate: what it actually cost, what the comparison revealed about hidden behavior, and a revised estimate for the remainder. The tranche found eleven undocumented consumers of the availability logic and cost 40 percent more than planned. The revised whole-programme estimate came out at €9.3 million, materially higher than any previous proposal. It was approved, because for the first time the number was derived from having done a piece of the work rather than from analysis.

The gate mechanism proved itself two tranches later. A stage intended to migrate the pricing engine reported at its gate that the approach was not working: the pricing rules were entangled with the promotions engine in ways that made an independent extraction impractical, and the tranche had consumed its budget establishing this. The recommendation was to stop that line and re-sequence, taking promotions first. The sponsor accepted it. Two things followed. The re-sequenced approach worked, and the fact that a gate had been used to stop something without anyone being blamed meant that subsequent gate reports were markedly more candid — one later tranche flagged a cost overrun at its midpoint rather than at its gate, which allowed a correction that would otherwise have arrived a quarter too late.
