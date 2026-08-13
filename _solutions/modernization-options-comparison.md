---
title: Modernization Options Comparison
description: Present retire, keep, encapsulate, replace, and rewrite as costed alternatives side by side, instead of asking approval for one preferred answer.
category:
- Architecture
- Management
- Business
problems:
- modernization-roi-justification-failure
- modernization-strategy-paralysis
- difficulty-quantifying-benefits
- second-system-effect
- technology-lock-in
- obsolete-technologies
- system-stagnation
- decision-paralysis
- budget-overruns
- high-maintenance-costs
- accumulated-decision-debt
- premature-technology-introduction
- competitive-disadvantage
- legacy-skill-shortage
- technology-stack-fragmentation
- vendor-dependency
- vendor-dependency-entrapment
layout: solution
---

## Description

A modernization options comparison presents the realistic alternatives for a legacy system — retire it, keep it as it is, encapsulate it behind an interface, replace it with a product, rewrite it, or migrate it largely unchanged — as costed, risk-assessed alternatives evaluated against the same criteria. The usual practice is different: a team decides internally which option it prefers, builds a case for that one, and presents it for approval. That framing puts the decision-maker in the position of either accepting or rejecting a single proposal, with no basis for judging whether the preferred option is the right one. It also removes their agency, which reliably produces either rejection or a demand for more analysis. Presenting alternatives changes the conversation from whether to approve engineering's answer into which trade-off the organization wants to make, which is a decision the organization is actually equipped to take.

## How to Apply ◆

> Two of the options are systematically underexamined: retiring the system entirely, and deliberately keeping it as it is — and one of them is frequently correct.

- **Include retire and keep as genuine candidates**, assessed with the same seriousness as the others. Usage data occasionally shows a system serves far less than assumed, and "keep it, contain it, and spend the money elsewhere" is a legitimate outcome that a comparison can reach and a single-option proposal never can.
- **Use the same criteria for every option**: cost to reach the end state, resulting run cost, risk during transition, risk of the end state, time to first benefit, and what it forecloses. Options assessed against different criteria cannot be compared, and an inconsistent table is the fastest way to look like advocacy.
- **Cost the do-nothing option too.** Continuing as-is is not free, and comparing three change options against an implicit zero is the single most common flaw in these documents. The cost of delay for the current state is the right figure here.
- **State the confidence of each estimate**, not just the number. A rewrite estimate deserves a much wider range than an encapsulation estimate, and hiding that difference behind two equally precise-looking numbers misleads the reader about the real risk.
- **Assess time to first benefit separately from total cost.** An option that costs more but delivers something in four months is often preferable in an organization that has to defend the spend annually, and this dimension is routinely omitted.
- **Note what each option forecloses.** Encapsulation preserves the option to replace later; a rewrite commits. Optionality has real value under uncertainty and belongs in the comparison explicitly.
- **Make a recommendation, with reasoning.** A comparison that refuses to recommend abdicates the expertise the team was asked for. The point is that the recommendation is visibly derived from the comparison rather than preceding it.
- **Show the hybrid.** The realistic answer for a large legacy estate is usually different options for different parts — retire two modules, encapsulate three, replace one. Presenting only whole-system options misrepresents the actual choice.
- **Get the estimates reviewed by someone with no stake**, before presenting. The most common failure is that the preferred option's estimate is optimistic and the alternatives' are not, which is rarely deliberate and always detected.

## Tradeoffs ⇄

> Comparing options produces better decisions and far better credibility, at the cost of estimating work for paths that will not be taken.

**Benefits:**

- The decision-maker can weigh trade-offs rather than accept or reject a single proposal, which is both a better decision process and much more likely to end in approval.
- The team's credibility rises considerably, because a document that seriously assesses alternatives to its own recommendation does not read as advocacy.
- Retire and keep get genuine consideration, and one of them is the right answer more often than engineering-led proposals suggest.
- The comparison surfaces hybrids, which are usually the realistic answer for a system of any size.
- When the chosen option later proves difficult, the record of what was compared and why prevents the decision being relitigated from scratch.

**Costs and Risks:**

- Estimating options that will not be pursued is real effort with no direct return, and it delays the decision.
- Estimate quality varies enormously across options, and placing a well-founded encapsulation figure next to a speculative rewrite figure gives both a false appearance of comparability.
- More options can deepen paralysis rather than resolving it, particularly in organizations already prone to deferral.
- The comparison can be constructed, consciously or not, so that the preferred option wins — through criteria selection as much as through estimates.
- Presenting a genuine option the team considers wrong risks it being chosen, which is a real cost of the honesty the method demands.

## How It Could Be

A logistics company's warehouse management system was the subject of a rewrite proposal estimated at €6 million over two years. The board declined it twice without a stated reason. On the third attempt the team presented five costed options instead of one. Continuing as-is came with a cost of delay of about €95,000 a month, rising. Encapsulation behind a service layer: €1.4 million, eighteen months to complete, first benefit in four months, preserves the option to replace later. Package replacement: €3.2 million with a wide range, high transition risk, a fit assessment showing two of eleven required capabilities unsupported. Rewrite: €6.1 million with a very wide range, first benefit in roughly twenty months. Retire and absorb into the group's other warehouse system: €2.1 million but requiring a business decision about site consolidation that was not engineering's to make. The board chose encapsulation, and separately opened the consolidation question that the retire option had surfaced.

The comparison's most valuable output was one nobody had expected to include. Assessing the retire option required someone to ask which sites the system actually served, which produced the finding that two of the seven had been migrated to the group system three years earlier and that the legacy system was still running interfaces for them — maintained, monitored, patched, and used by nothing. Those interfaces were decommissioned within a month, independent of any option being chosen. The team's estimate of the recurring saving was roughly €140,000 a year, found as a side effect of taking seriously an option they had assumed was not viable.
