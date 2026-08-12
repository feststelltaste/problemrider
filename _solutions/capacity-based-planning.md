---
title: Capacity Based Planning
description: Derive commitments from measured historical throughput rather than from desired dates, and express them as ranges with stated confidence.
category:
- Management
- Process
problems:
- unrealistic-deadlines
- unrealistic-schedule
- missed-deadlines
- planning-credibility-issues
- reduced-predictability
- constantly-shifting-deadlines
- large-estimates-for-small-changes
- deadline-pressure
- cascade-delays
- increased-time-to-market
- staff-availability-issues
- overworked-teams
- time-pressure
- delayed-project-timelines
- competing-priorities
- extended-cycle-times
- increased-stress-and-burnout
- increased-technical-shortcuts
- market-pressure
- mental-fatigue
- priority-thrashing
- team-demoralization
- uneven-work-flow
- uneven-workload-distribution
- budget-overruns
- changing-project-scope
- poor-planning
- project-resource-constraints
- scope-change-resistance
- stakeholder-confidence-loss
- stakeholder-dissatisfaction
layout: solution
---

## Description

Capacity based planning derives what a team can commit to from what it has actually delivered, rather than from what a date requires it to deliver. It rests on two shifts. The first is measurement: throughput and cycle time are recorded over a meaningful history and used as the basis for forecasts, replacing per-task estimates aggregated into a plan. The second is honesty about uncertainty: commitments are expressed as ranges with confidence levels rather than as single dates that everyone privately knows are optimistic. In legacy systems this matters more than elsewhere, because the dominant source of schedule variance is not the work anyone planned — it is the unplanned discovery that a change touches an undocumented dependency, and no amount of upfront estimation predicts that. Historical throughput, however, already contains that variance, because it was present in every past period too.

## How to Apply ◆

> Legacy work has a long tail of surprises; a planning method that treats the tail as an exception will be wrong every time, while one that measures it as a normal property of the system will not.

- Start by **measuring actual throughput** for at least eight to twelve completed periods: how many items of what type were finished, and how long each took from start to done. Use whatever units the team already works in. The absolute numbers matter far less than their spread.
- Forecast with **ranges derived from that history**, not with a single number. State plans as "eighty-five percent confident by the end of March, fifty percent confident by mid-February." The spread of past performance is the honest measure of uncertainty, and stating it converts an argument about optimism into a discussion about acceptable risk.
- Subtract **known non-project time before committing**, not after. Support rotations, incident response, meetings, holidays, onboarding, and the interrupt load of a legacy system are not overhead to be absorbed heroically — they are capacity that does not exist. Teams that plan at one hundred percent of nominal capacity miss dates structurally, not occasionally.
- Track and publish the **interrupt load** as a separate figure. In many legacy teams it accounts for thirty to fifty percent of capacity, and it is usually invisible in planning because it is invisible in the plan. Making it a number changes both the forecast and, eventually, the investment decisions that would reduce it.
- When a date is fixed externally, **vary scope rather than the forecast**. Compute what fits in the available capacity at a stated confidence, and present that as the deliverable. Committing to a scope the capacity does not support does not create capacity; it defers the discovery of the shortfall to the point where the fewest options remain.
- Use **reference class comparison** for large or unfamiliar work: find the three most similar things the team has completed and how long they actually took. This is a far better predictor than decomposition-and-estimation, which systematically omits the work nobody thought of — which in legacy systems is most of it.
- **Re-forecast on a fixed cadence** using updated actuals, and treat a moving forecast as information rather than failure. A plan that never changes in a legacy environment is a plan that has stopped tracking reality.
- Record **estimate versus actual** for a sample of work and review it quarterly. The purpose is calibration, not accountability; if the review is used to criticize individuals, the estimates will adjust to protect their authors and the data becomes worthless.
- Present forecasts to stakeholders with the **evidence attached** — the throughput history, the confidence basis, the assumed interrupt load. A forecast that arrives as a bare date invites negotiation; one that arrives with its derivation invites a discussion about which assumption to change.

## Tradeoffs ⇄

> Capacity based planning produces forecasts that are considerably more accurate and considerably less welcome, because it removes the optimism that made previous plans acceptable.

**Benefits:**

- Forecast accuracy improves substantially, because historical throughput already includes the interruptions, rework, and surprises that per-task estimates systematically exclude.
- Planning credibility recovers over time, since dates that are met build trust faster than ambitious dates that are missed.
- The real cost of the interrupt load becomes visible and quantified, which is usually the prerequisite for any investment in reducing it.
- Deadline pressure declines because commitments are derived from evidence rather than negotiated downward, which removes the structural overcommitment that drives sustained overtime.
- Scope conversations happen early, when adjusting scope is cheap, rather than in the final weeks when it is not.

**Costs and Risks:**

- The first honest forecasts are usually much later than what the organization has been told, and delivering that message is politically costly regardless of how well the evidence is presented.
- Ranges and confidence levels are unfamiliar to many stakeholders, who may hear "eighty-five percent confident by March" and record "March." Consistent, patient communication is required, and it takes several cycles.
- The method needs a meaningful history, so newly formed teams or teams entering an unfamiliar subsystem have little to work from initially.
- Historical throughput assumes continuity. A significant change in team composition, technology, or the nature of the work invalidates the baseline, and legacy modernization efforts often involve exactly such changes.
- Measurement can be corrupted if throughput becomes a target. Teams optimize what is counted, typically by splitting work finer, which inflates the metric without improving delivery.

## How It Could Be

A team maintaining a telecommunications billing system had missed eleven of its last twelve quarterly commitments, and planning had degenerated into a ritual where everyone privately doubled the stated dates. They spent two weeks reconstructing actual throughput from their ticket system and found that they had completed between fourteen and twenty-six items per month over the previous year, with support work consuming an unmeasured but substantial share. Their next commitment was presented as a range with confidence levels, and separately quantified the support load at roughly thirty-eight percent of capacity. The immediate reaction from their director was that the forecast was unacceptable. The subsequent conversation, however, was about the thirty-eight percent — which nobody had known — and led to funding a dedicated support engineer. The team met its next three quarterly commitments.

A second team faced a fixed regulatory deadline nine months out and was asked whether the full compliance scope would fit. Rather than answering yes under pressure, they used reference class comparison against two previous compliance efforts and forecast that roughly seventy percent of the scope fit at eighty-five percent confidence. They presented the seventy percent as the committed deliverable and the remaining thirty percent as ranked items that would land only if the earlier work went unusually well. Two of the deferred items were subsequently found to be unnecessary on closer reading of the regulation, and the mandatory scope was delivered six weeks before the deadline — the first time in that organization's memory that a compliance project had not required a crisis push at the end.
