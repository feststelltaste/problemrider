---
title: Defect Triage Process
description: Assess every reported defect against stated criteria at a regular cadence,
  classify its cause, and use the accumulated classification to fix categories rather
  than instances.
category:
- Process
- Code
- Testing
problems:
- partial-bug-fixes
- delayed-bug-fixes
- increased-bug-count
- quality-degradation
- high-defect-rate-in-production
- constant-firefighting
- delayed-issue-resolution
- regression-bugs
- quality-compromises
- brittle-codebase
- reduced-feature-quality
- workaround-culture
- avoidance-behaviors
- blame-culture
- increased-risk-of-bugs
- increasing-brittleness
- negative-brand-perception
- user-trust-erosion
layout: solution
related_solutions:
- slug: explicit-prioritization-framework
  similarity: 0.65
- slug: workaround-registry
  similarity: 0.65
- slug: code-hotspot-analysis
  similarity: 0.65
- slug: blameless-postmortems
  similarity: 0.65
- slug: debt-classification
  similarity: 0.65
- slug: debt-accrual-analysis
  similarity: 0.65
---

## Description

A defect triage process is a regular, short review in which newly reported defects are assessed against stated criteria — severity, affected users, whether a workaround exists — assigned an owner and a priority, and classified by underlying cause. The classification is the part teams skip and the part that matters most. Handling defects one at a time, in the order they arrive or the order they are shouted about, means fixing symptoms forever: the same class of defect recurs because nobody has looked across instances to see that thirty of them share one cause. Legacy systems generate defects faster than any team can fix them individually, so the only tractable strategy is to fix categories. Triage is the mechanism that turns a stream of individual reports into the data needed to identify those categories.

## How to Apply ◆

> In a system producing more defects than the team can fix, the decision about what not to fix is being made either explicitly by triage or implicitly by whoever complains loudest.

- Hold triage on a **fixed, frequent cadence** — twice a week for a high-volume system — and keep it short. A long, infrequent triage meeting accumulates a backlog that is too large to assess properly, so items get waved through with a guess.
- Use **written severity criteria** rather than judgment in the moment. Data corruption, security exposure, blocked business process, degraded experience, and cosmetic are enough. Without written criteria, severity tracks who reported it.
- Record **whether a workaround exists and what it costs**. A high-severity defect with a cheap workaround may reasonably wait behind a medium one without any; without this field the priority decision is made on severity alone and is frequently wrong.
- **Classify the cause, not just the symptom** — missing validation, unhandled null, race condition, configuration error, misunderstood requirement, regression from another change. Use a small fixed taxonomy of eight to twelve categories so the data stays comparable.
- **Review the classification distribution quarterly.** This is where the value is. One dominant category is a systemic finding, and fixing it addresses defects that have not been reported yet. Individual triage decisions matter far less than this aggregate.
- Assign **one owner per accepted defect** at triage, not later. Defects with no owner age indefinitely, and defect age is the metric that best predicts whether something will ever be fixed.
- **Decide explicitly not to fix** where that is the answer, and record why. An unfixed defect sitting open forever is worse than a closed one with a stated reason, because it pollutes the data and gives false hope to whoever reported it.
- Require that a fix addresses the **cause rather than the instance**, or that the partial nature of the fix is recorded. Partial fixes are sometimes correct under time pressure; they become a problem only when nobody notes that the underlying cause remains.
- **Track regressions separately.** A defect introduced by a recent change is a different signal from one that has been latent for years, and a rising regression rate points at the test suite rather than at the code.
- Feed the classification data into the **improvement budget and hotspot analysis**. The categories that dominate the distribution are the best available evidence for what to invest in.

## Tradeoffs ⇄

> Triage makes prioritization explicit and produces the data needed to fix causes, at the cost of recurring meeting time and a classification discipline that decays easily.

**Benefits:**

- Prioritization moves from social pressure to stated criteria, which is more defensible and produces better outcomes for users who are not good at escalating.
- Cause classification identifies systemic problems that are invisible in individual reports, which is the only way to reduce defect volume rather than keep pace with it.
- Defects get owners immediately, which is the single strongest predictor of whether they get fixed.
- Deliberate non-fix decisions clear the backlog of items that were never going to be addressed, making the remaining list meaningful.
- Regression trends give early warning about test coverage erosion, usually well before it shows up as a production incident.

**Costs and Risks:**

- Recurring meeting time for several people, which is a real cost and is often the first thing dropped when the team is busy — precisely when defect volume is highest.
- Classification degrades toward whatever category is easiest to select, and once the data is unreliable the aggregate analysis becomes misleading rather than merely useless.
- Written criteria get gamed. Reporters learn which words produce high severity, and the criteria need occasional recalibration.
- Triage can become a bottleneck if the meeting is the only path to a decision, delaying genuinely urgent items that should bypass it.
- Explicitly declining to fix defects is politically uncomfortable and can damage relationships with the people who reported them, even when it is the right decision.

## How It Could Be

A team maintaining a retail point-of-sale system received 60 to 90 defect reports a month and fixed roughly 40, choosing them by a mixture of severity and who was asking. They introduced twice-weekly triage with five written severity levels, a workaround field, and a nine-category cause taxonomy. The first quarterly review of the classification data showed that 38 percent of all defects fell into one category: unhandled edge cases in date and time handling, spread across eleven different modules. Nobody had seen this, because each instance had been fixed locally by whoever picked it up. The team built a shared date-handling module and migrated the eleven call sites over two months. Reports in that category fell from an average of 26 a month to 3.

The explicit non-fix decision changed the backlog more than any fix did. Triage worked through 340 open defects over six weeks and closed 190 of them as deliberately not-fixed, each with a stated reason: superseded, no longer reproducible, affecting a feature scheduled for removal, or judged not worth the cost. Nineteen were reopened by their reporters, and four of those turned out to be genuinely important and were fixed. The remaining 150 open defects were, for the first time, a list the team actually intended to work through, which meant the age of the oldest open defect became a metric worth reporting rather than a source of embarrassment.
