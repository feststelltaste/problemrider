---
title: Baseline Measurement
description: Measure the current state before you change it, because a benefit that
  has no "before" can never be demonstrated afterwards.
category:
- Process
- Management
- Business
problems:
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- invisible-nature-of-technical-debt
- planning-credibility-issues
- high-maintenance-costs
- maintenance-cost-increase
- short-term-focus
- quality-degradation
- increasing-brittleness
- slow-development-velocity
- resource-waste
- wasted-development-effort
- budget-overruns
- declining-business-metrics
- deployment-risk
- increased-cost-of-development
- legacy-system-documentation-archaeology
- poor-planning
- reduced-predictability
- regulatory-compliance-drift
- stakeholder-confidence-loss
- stakeholder-frustration
- high-technical-debt
layout: solution
related_solutions:
- slug: delivery-performance-metrics
  similarity: 0.75
- slug: benefits-realization-tracking
  similarity: 0.7
- slug: outcome-based-goal-setting
  similarity: 0.65
- slug: quality-ratchet
  similarity: 0.65
- slug: fast-feedback-loops
  similarity: 0.65
- slug: improvement-budget
  similarity: 0.65
---

## Description

Baseline measurement is the discipline of recording the current state — in numbers, before the work starts — of whatever the work is supposed to improve. It is the cheapest possible intervention against the most common reason technical improvements cannot be justified: not that their benefits are unreal, but that nobody can demonstrate them, because no one wrote down what things were like beforehand. The pattern is consistent. A team spends a quarter reducing build times, incident rates, or manual effort, achieves a genuine improvement, and then cannot say by how much, because the "before" exists only as a shared impression. The next proposal is then met with the same scepticism as the last, and the cycle continues. Measuring first costs days; not measuring first costs the credibility that funds everything afterwards.

## How to Apply ◆

> The measurement usually has to happen before anyone has agreed to fund the work, which means the team has to do it speculatively — and that is exactly why it does not happen.

- **Decide what the work is supposed to change**, in one sentence, before choosing what to measure. Improvements that cannot name the measure they intend to move are improvements whose value will be contested afterwards, correctly.
- **Reconstruct history rather than starting a clock.** Version control, ticket systems, deployment logs, and incident records usually contain enough to reconstruct six to twelve months retrospectively. This is far better than a baseline starting today, because it shows the trend as well as the level.
- **Capture three to five measures, no more**, and pick ones that are cheap to repeat. A baseline that takes two weeks to produce will be measured once, and a single measurement is not a baseline — the comparison is the point.
- **Measure the distribution, not just the average.** The median build time and the ninety-fifth percentile tell different stories, and improvements frequently move one without the other. Recording both prevents an argument later about which one counted.
- **Include a measure the business already cares about**, even if it is only loosely coupled to the work: time to fulfil a request, error rate visible to customers, hours of manual effort in a department. A purely technical baseline proves an improvement to an audience that was never in doubt.
- **Write down the conditions**, not only the numbers: team size, system load, the release cadence at the time, anything unusual in the period. Baselines are attacked afterwards on the grounds that something else changed, and the record is the defence.
- **Publish the baseline before the work starts**, ideally to the people who will judge the result. A baseline produced after the fact, however honestly, invites the suspicion that it was chosen to flatter the outcome.
- **Re-measure at agreed points**, not only at the end. Interim measurements catch an intervention that is not working while there is still time to change course, which is worth more than the eventual proof.
- **Report honestly when the number did not move.** A team that reports its failures is believed when it reports its successes, and this is the whole mechanism by which measurement builds the credibility that later proposals depend on.

## Tradeoffs ⇄

> Baselines are cheap and are what makes benefits provable, but they take effort before anything is funded and they create the possibility of being visibly wrong.

**Benefits:**

- Benefits become demonstrable rather than asserted, which is the difference between a proposal that is believed next time and one that is not.
- Interim measurement catches ineffective interventions early, before the full investment is spent on an approach that is not working.
- The trend from reconstructed history is often a stronger argument than the level, since it shows where the situation is heading.
- Attributing improvement to a specific piece of work becomes possible, which is what converts a one-off approval into ongoing funding.
- The act of choosing measures forces clarity about what the work is actually for, which frequently changes what the team decides to do.

**Costs and Risks:**

- It takes effort before anything is approved, and that effort is uncompensated if the proposal is declined.
- A baseline creates the possibility of demonstrating that an improvement did not work, which is a real risk to the team that produced it and a strong incentive not to measure.
- Measures become targets. Anything used to judge success will eventually be optimized directly, sometimes at the expense of what it was proxying for.
- Attribution is contestable: other things change during the same period, and a determined sceptic can always propose an alternative explanation.
- A poorly chosen measure locks the work into optimizing the wrong thing, and changing the measure mid-way looks like moving the goalposts even when it is correct.

## How It Could Be

A team spent a quarter reducing their build and test cycle, and reported afterwards that it was "much faster now." Asked how much faster, they could not say — the previous duration existed only as a shared memory of "about half an hour." The improvement, which was substantial, produced no change in how the organization funded such work. Before the next effort they spent two days reconstructing twelve months of pipeline durations from their CI system: median 31 minutes rising to 38 over the year, ninety-fifth percentile 74 minutes. They also recorded two business-facing measures: the median time from a defect being reported to a fix reaching production, and the number of changes deployed per month. After the work, the same four numbers were re-measured. Median pipeline duration 8 minutes, ninety-fifth percentile 14, defect-to-production time down from 9 days to 3, deployments up 60 percent. The last two were what secured the following year's improvement budget; the first two only explained them.

The honesty rule was tested a year later. The team invested six weeks in a caching layer expected to cut a report generation time that a business department complained about constantly. The baseline said median 42 seconds, ninety-fifth percentile 6 minutes. After the work: median 4 seconds, ninety-fifth percentile 5 minutes 40. The tail — which was what the department actually complained about — had barely moved, because it was dominated by a query the cache did not touch. The team reported this plainly rather than leading with the median. The immediate cost was an uncomfortable meeting. The lasting effect was that when the same team said, eight months later, that a proposed change would deliver a specific improvement, the number was accepted without argument.
