---
title: Debt Accrual Analysis
description: Find out why debt keeps appearing in the same places, and fix the mechanism — because paying down debt while the accrual continues is a treadmill.
category:
- Process
- Code
- Management
problems:
- high-technical-debt
- accumulation-of-workarounds
- increased-technical-shortcuts
- increasing-brittleness
- quality-degradation
- workaround-culture
- invisible-nature-of-technical-debt
- quality-compromises
- refactoring-avoidance
- copy-paste-programming
- inconsistent-execution
- maintenance-cost-increase
- convenience-driven-development
- short-term-focus
- code-duplication
layout: solution
---

## Description

Debt accrual analysis asks a different question from the usual one. Instead of "what debt do we have," it asks "what keeps producing it," and treats the answer as the thing to fix. The distinction matters because remediation without it is a treadmill: a team pays down debt at some rate while the organization generates it at another, and if the second rate is higher, the effort is invisible and the team's morale eventually gives out. The causes are rarely mysterious and almost never a lack of skill or care. They are structural — deadline pressure applied at a specific point in the cycle, a missing test capability that makes the safe path expensive, an ownership gap where nobody is responsible, a review practice that does not catch a particular class of problem. Each is addressable, and addressing one typically prevents more debt than months of remediation removes.

## How to Apply ◆

> Legacy debt is usually not one historical event but an ongoing process, and the process is normally visible in the version control history if anyone looks.

- **Start from the recent debt, not the old debt.** Take the items introduced in the last six to twelve months — identifiable from the workaround registry, review comments, and commit history — and analyze those. Debt from 2011 tells you about an organization that no longer exists.
- **Look for clustering in time, place, and circumstance.** Debt concentrated in one subsystem points at that subsystem's structure or ownership; debt concentrated in the weeks before releases points at the release process; debt concentrated in one team's work points at capability or workload.
- **Ask what the cheap path was at the moment of decision.** Debt is almost always the rational local choice under the constraints in force. The productive question is what made the good path expensive — no tests, no time, no knowledge, no authority to say no — because that constraint is the actual target.
- **Use blameless technique.** The analysis names mechanisms, not people. The moment it identifies individuals, the information stops flowing and the analysis becomes worthless, because the people who know why the shortcut was taken are the ones who took it.
- **Look for the missing capability.** A recurring pattern of untested changes in one area frequently means testing that area is genuinely hard, not that developers are careless. The intervention is then a seam or a test fixture, not an exhortation.
- **Check the incentives.** If delivery is measured and quality is not, debt accrual is the predictable result and no amount of process will fix it. This finding is uncomfortable and is often the real one.
- **Quantify the accrual rate** where possible — workarounds added per quarter, shortcuts recorded per release. A rate makes it possible to tell whether an intervention worked, and without one the improvement is a matter of opinion.
- **Fix one mechanism at a time and re-measure.** Several simultaneous interventions make it impossible to know which worked, and the knowledge of which mechanisms matter is the durable output.
- **Feed the findings into retrospectives** rather than producing a report. The team that generates the debt is the one that has to change something, and a report handed to them will not do it.

## Tradeoffs ⇄

> Fixing the mechanism prevents more debt than remediation removes, but the causes are often organizational and outside the team's control to change.

**Benefits:**

- The accrual rate falls, which is the only way a remediation effort ever gets ahead rather than treading water.
- Interventions target constraints rather than behavior, which works — telling people to write better code under unchanged constraints reliably does not.
- The analysis frequently reveals a missing capability whose absence was invisible, such as an area where testing is genuinely impractical.
- Morale improves when the team can see the rate changing, rather than remediating indefinitely against an unmeasured inflow.
- The findings are usually cheap to act on relative to the debt they prevent, since a process or tooling fix is small compared to months of remediation.

**Costs and Risks:**

- The causes are frequently organizational — deadline pressure, incentives, resourcing — and naming them does not give the team power to change them.
- The findings can be politically uncomfortable, particularly when the honest answer is that management pressure is the mechanism.
- Attribution is difficult: debt introduced now may only become visible in a year, so the analysis always lags the behavior it examines.
- Without a blameless frame the analysis rapidly becomes an audit, and the information it needs stops being available.
- Measuring the accrual rate requires the recording discipline — workaround registry, marked shortcuts — that may not exist yet, so the measurement work comes first.

## How It Could Be

A team had run a technical debt reduction effort for three quarters and could not demonstrate that the situation had improved. They analyzed the debt introduced during those same three quarters and found 34 new items, against roughly 40 remediated — they had been running at close to break-even without knowing it. The clustering was stark: 21 of the 34 had been introduced in the final two weeks before a release, and 19 of those were in code paths that had no test coverage. The mechanism was not carelessness. Their release process concentrated integration work into a two-week window, and in that window the safe path — writing a characterization test before changing untested code — cost a day that nobody had. Two interventions followed: continuous integration to spread the work out, and a small library of test fixtures for the three most awkward areas. Debt introduced in the following two quarters fell to 11 items.

The incentive finding came from the same analysis and was harder to act on. Of the 13 items not associated with the release window, 9 traced back to a single recurring situation: a request arriving directly from a senior stakeholder to a developer, bypassing the backlog, with an implied urgency. Nobody had ever refused one. The team's engineering manager took this to her director not as a complaint but as a measured finding — nine debt items in nine months from one identifiable pathway — and the outcome was a rule routing such requests through her. The rule was broken twice in the following year rather than roughly monthly, which the team regarded as a success rather than a failure of the rule.
