---
title: Outcome-Based Goal Setting
description: State goals as changes in the world with a measure attached, not as lists of things to build, so that delivery can be judged by effect rather than by volume.
category:
- Business
- Management
- Process
problems:
- unclear-goals-and-priorities
- feature-factory
- declining-business-metrics
- short-term-focus
- product-direction-chaos
- competitive-disadvantage
- delayed-value-delivery
- wasted-development-effort
- gold-plating
- feedback-isolation
- stakeholder-dissatisfaction
- reduced-innovation
- planning-dysfunction
- competing-priorities
- eager-to-please-stakeholders
- feature-creep
- individual-recognition-culture
- market-pressure
- micromanagement-culture
- perfectionist-culture
- planning-credibility-issues
- priority-thrashing
- project-authority-vacuum
- stakeholder-frustration
- team-demoralization
- unmotivated-employees
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- process-software-misfit
layout: solution
---

## Description

Outcome-based goal setting means expressing what a team is trying to achieve as a change in observable reality with a measure attached — support contacts for this process reduced by half, month-end close completed in one day instead of four — rather than as a list of features to deliver. The distinction sounds semantic and is not. A team given a feature list can only succeed by delivering it, so it will deliver it whether or not it helps, and nobody will ever find out. A team given an outcome can succeed by any means, including by discovering that the requested feature was the wrong approach and doing something cheaper. In legacy contexts this matters especially because the cheapest path to most outcomes is often not a new feature at all: it is fixing something that is broken, removing something that is confusing, or making something faster.

## How to Apply ◆

> Legacy modernization is particularly vulnerable to output-based goals, because "migrate these forty modules" is measurable, achievable, and can be completed without anything getting better.

- **State each goal as a change with a number and a date.** "Reduce the time to onboard a new corporate customer from nine days to two by the end of Q3" is a goal. "Build the customer onboarding portal" is a plan, and possibly the wrong one.
- Keep the number of goals **very small** — two or three per team per quarter. A list of eight outcomes is a list of eight priorities, which is a list of none, and the team will fall back to picking by pressure.
- **Establish the baseline before committing to the target.** A goal to halve something whose current value nobody has measured cannot be assessed, and this is the most common way outcome goals quietly revert to output goals.
- **Let the team choose the means.** This is the entire mechanism. A goal that arrives with the solution attached is an output goal wearing outcome language, and it forecloses the cheaper alternatives the team is best placed to find.
- Include **health and risk outcomes**, not only growth ones — incident hours, time to restore, the share of capacity going to unplanned work. Without these, the framework becomes a way of directing all capacity at business metrics while the system degrades underneath.
- **Review progress on a cadence** and treat a missed goal as information rather than failure. Goals that are always achieved were set too low, and a framework in which missing one is punished produces conservative targets and defensive reporting.
- **Distinguish the goal from the commitments** made along the way. Some work is mandatory regardless of outcome — compliance, security patching, contractual obligations — and pretending it serves an outcome distorts both.
- **Record what was tried and did not work.** The largest benefit of outcome framing is learning which interventions move which measures, and that learning is lost if only successes are documented.
- **Connect goals to what the team can actually influence.** A goal tied to a measure the team cannot move through its own work produces cynicism, and correctly so.

## Tradeoffs ⇄

> Outcome goals redirect effort from volume to effect, but they require measurement, tolerance for uncertainty, and stakeholders willing to give up specifying the solution.

**Benefits:**

- Effort goes to what changes the measure, which frequently turns out to be smaller and cheaper than the feature that was originally requested.
- The feature-factory pattern is interrupted at its source, since shipping volume stops being the definition of success.
- Teams gain genuine autonomy over how to achieve results, which is among the strongest drivers of motivation and retention.
- Unsuccessful approaches become visible and can be abandoned, rather than being completed because they were on the plan.
- Maintenance and risk-reduction work can compete on equal terms when health outcomes are included, because it often moves those measures better than features do.

**Costs and Risks:**

- It requires measurement infrastructure that many legacy systems lack, and instrumenting them is itself a project.
- Outcomes are influenced by factors outside the team's control, which makes attribution contestable and creates room for both unfair blame and undeserved credit.
- Stakeholders who are used to specifying solutions often experience outcome framing as evasion, and the transition needs sustained explanation.
- Measures get gamed. Any number that determines how a team is judged will eventually be optimized directly, sometimes at the expense of the thing it was proxying for.
- Goals stated for a quarter can be too short a horizon for legacy modernization, where meaningful outcomes may take a year, and forcing quarterly outcomes onto such work produces artificial milestones.

## How It Could Be

A team maintaining an insurance policy administration system was given a quarterly plan consisting of eleven features, of which they delivered nine. Nobody could say afterward whether anything had improved. The following quarter their director reframed the work as two outcomes: reduce the average time to issue a commercial policy from six days to three, and reduce policy-related support contacts by a third. The team's first action was not to build anything — it was to sit with two underwriters for a day. They found that four of the six days were spent waiting for a document check that was already automated but whose result was not shown anywhere in the interface. A two-day change surfaced it. Average issuance time fell to two and a half days, and seven of the eleven originally planned features were never built because the outcome had been reached without them.

The health outcomes changed a longer-running argument. The same team added a third goal: reduce the share of capacity consumed by unplanned work from an established baseline of 41 percent to under 30. This gave maintenance work a target to be measured against, which it had never had. Over two quarters they addressed the two subsystems generating most of the incident load and reached 26 percent. The freed capacity — roughly a day and a half per person per week — was what made the next quarter's business outcomes achievable, and the connection between the two was visible in the numbers rather than being an assertion the team had to keep making.
