---
title: Debt Remediation Estimation
description: Put an effort figure on each debt item so the total becomes a finite number — because an unsized problem cannot be planned and feels infinite.
category:
- Code
- Management
- Process
problems:
- high-technical-debt
- invisible-nature-of-technical-debt
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- maintenance-paralysis
- modernization-strategy-paralysis
- large-estimates-for-small-changes
- planning-credibility-issues
- refactoring-avoidance
- analysis-paralysis
- budget-overruns
- accumulation-of-workarounds
- fear-of-change
- brittle-codebase
- poor-test-coverage
- core-modification-of-standard-software
layout: solution
---

## Description

Debt remediation estimation puts a rough effort figure on each known debt item and, from those, a total. The number matters less than the fact of having one. An unsized problem is experienced as infinite, and an infinite problem produces two responses, both bad: paralysis, because there is no point starting something that cannot be finished, and denial, because a problem that cannot be solved is easier not to look at. Both are common in legacy teams and both dissolve when the total turns out to be a number. The estimates do not need to be accurate — they need to be honest about their own uncertainty and consistent enough that items can be compared. Frequently the total is smaller than the dread suggested, and where it is genuinely large, at least the organization now knows what it is choosing not to do.

## How to Apply ◆

> The most useful number an assessment produces is not the total but the discovery that the three worst items account for most of it — and that they are individually finite.

- **Estimate in a coarse scale**, not in hours: a few days, a couple of weeks, a month or two, a quarter or more. Precision here is false, and coarse buckets are faster to produce, easier to agree, and honest about what is known.
- **Estimate only the debt worth remediating.** Sizing all 187 items in a backlog is a waste; sizing the interest-bearing subset takes a fraction of the time and is what informs the decision.
- **Estimate the smallest safe increment**, not the ideal end state. "What would it take to stop this hurting" is usually a fraction of "what would it take to do this properly," and conflating them is why debt items acquire estimates that guarantee they will never be approved.
- **Include the safety net in the estimate.** Remediating legacy code usually requires characterization tests first, and estimates that omit this are wrong by a large factor. This is the most common single reason debt estimates are exceeded.
- **State a range with the reasoning**, and be explicit where you do not know. An item estimated at "two weeks to two months, because we do not know how many consumers depend on this interface" invites the cheap investigation that would narrow it — which is often the right next step.
- **Attach the estimate to the cost it removes.** An item costing four days a month to live with and two weeks to fix pays back in under four months; the same fix against a dormant item never pays back. The pair of numbers is what makes the case, not either alone.
- **Publish the total and its distribution.** A total of "roughly eight to fourteen developer-months, of which two items account for half" is a management statement. It also frequently reveals that the situation is less catastrophic than everyone assumed.
- **Re-estimate after each remediation** using what it actually cost. Legacy remediation estimates improve dramatically after a few real data points, and the early ones are usually optimistic by a consistent factor worth measuring.
- **Do not let the estimate become a commitment.** These are sizing figures for prioritization, and treating them as delivery promises will cause the team to inflate them until they stop being useful.

## Tradeoffs ⇄

> Sizing turns an unbounded dread into a finite plan, at the cost of estimation effort and estimates that will be wrong in ways that can be held against the team.

**Benefits:**

- The problem becomes finite, which is the precondition for planning it, funding it, or consciously deciding not to.
- Payback becomes computable when the estimate is paired with the ongoing cost, which is what lets debt work compete on evidence.
- The distribution usually shows that a small number of items dominate the total, which turns an overwhelming list into a short one.
- Paralysis and denial both weaken, because there is now something to start rather than an endless condition to endure.
- Estimation accuracy improves over time as real remediation data accumulates, which improves every subsequent plan.

**Costs and Risks:**

- Legacy remediation estimates are genuinely unreliable, because the work regularly uncovers dependencies nobody knew about — which is the nature of the thing being estimated.
- Estimates get treated as commitments regardless of how they are labelled, and the team pays for the overrun.
- Sizing takes time from people who are already the constraint, and it produces no working software.
- A large honest total can confirm an organization's belief that the situation is hopeless, producing the opposite of the intended effect.
- Estimating the smallest safe increment can understate what will eventually be needed, leaving a sequence of partial remediations that never reaches a good state.

## How It Could Be

A team's technical debt was described in every planning discussion as overwhelming, and the phrase used internally was that the system would need "a rewrite eventually." They classified their backlog, found 24 interest-bearing items, and sized those over two days in four coarse buckets. The total came to roughly nine to sixteen developer-months. Two items — the duplicated pricing logic and the absence of any test around the settlement batch — accounted for about half of it. The reaction in the room was audible relief: nine to sixteen developer-months was a large number but a comprehensible one, and considerably smaller than the rewrite everyone had been half-assuming. The two dominant items were funded as a single piece of work over the following two quarters.

Pairing sizes with ongoing costs reordered the list in a way nobody expected. An item everyone wanted to fix — a badly structured reporting module, estimated at six weeks — turned out to cost roughly half a day a month to live with, a payback of about twenty years. An item nobody had advocated for, a missing index and a badly shaped query, was estimated at three days and was costing about three days a month in support handling of timeouts. It was done that week. The team's later summary of the exercise was that they had spent four years prioritizing debt by how much it annoyed them, and that the annoyance had almost no relationship to the cost.
