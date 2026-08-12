---
title: Small Change Batches
description: Keep every change small enough to be understood, reviewed, tested, and reverted as a single unit, and integrate it before starting the next one.
category:
- Process
- Code
- Team
problems:
- large-pull-requests
- extended-review-cycles
- review-bottlenecks
- long-lived-feature-branches
- superficial-code-reviews
- reduced-code-submission-frequency
- extended-cycle-times
- development-disruption
- delayed-issue-resolution
- fear-of-breaking-changes
- large-estimates-for-small-changes
- increased-bug-count
- author-frustration
- fear-of-failure
- inadequate-initial-reviews
- increased-time-to-market
- past-negative-experiences
- perfectionist-culture
- procrastination-on-complex-tasks
- reduced-predictability
- reduced-review-participation
- review-process-avoidance
- review-process-breakdown
- rushed-approvals
- team-members-not-engaged-in-review-process
layout: solution
---

## Description

Small change batches are a working discipline: each change that reaches integration is scoped so that one person can understand it end to end in a single sitting, and it is integrated before the next change begins. Batch size is the variable that quietly governs most of a team's flow metrics — review latency, defect escape rate, merge conflict frequency, and the confidence with which anyone can revert. It compounds badly: a large change takes longer to review, so it waits longer, so it diverges further from the mainline, so integrating it is riskier, so reviewers are more reluctant to engage with it. In legacy systems the pressure toward large batches is especially strong, because touching one part of a tangled codebase seems to require touching five others. Reducing batch size in that environment is therefore not primarily a process rule but a design skill: learning to split work that appears indivisible.

## How to Apply ◆

> Legacy code resists small changes because responsibilities are entangled; the techniques below are mostly about creating a safe way to land a partial change rather than about being more disciplined.

- Set an explicit, visible **size guideline** rather than a hard limit — for example, "a change should be reviewable in under thirty minutes." Expressing it as review time rather than lines changed keeps mechanical refactorings and generated code from being penalized while still catching genuinely large changes.
- **Separate refactoring from behavior change** into distinct commits or pull requests. A change that both moves code and alters what it does is disproportionately hard to review, because the reviewer cannot tell which diff lines are supposed to be behavior-neutral. Landing the refactoring first, verified by existing tests, makes the subsequent behavior change small and obvious.
- Use **feature toggles** to land incomplete work safely. A feature can be merged in five small increments behind a disabled toggle rather than accumulating on a branch for three weeks. This decouples "integrated" from "released," which is what makes small batches compatible with features that take a long time to build.
- Apply the **sprout and wrap techniques** when the existing code is too risky to modify directly: add the new behavior in a new function or class that the legacy code calls, rather than editing the legacy code in place. The change stays small and the new code is testable even when its surroundings are not.
- Land **preparatory changes independently**. If a change requires a new interface, an extracted method, or a widened parameter type, submit those first as standalone changes that leave behavior unchanged. Each is trivially reviewable, and the eventual functional change shrinks to the part that actually matters.
- Make **integration frequency** the metric the team tracks, not branch age. Ask in standups when each in-flight change will be integrated, not when the feature will be finished. Branches that have not been integrated for more than a couple of days are treated as a risk to be discussed, not as normal.
- Split by **vertical slice rather than by layer**. Delivering one narrow end-to-end path — one field, one record type, one customer segment — produces a small change that is independently valuable and testable, whereas splitting by layer produces small changes that are individually meaningless and must all land together anyway.
- When a change genuinely cannot be split, say so explicitly and **plan the review** instead of submitting it cold: walk the reviewer through the change in a short session, agree on which areas warrant close reading, and note the rest as reviewed-by-walkthrough. This is a fallback, and its frequency is worth tracking, because a team that uses it often has a structural problem rather than an unlucky change.

## Tradeoffs ⇄

> Small batches reduce the risk and latency of each individual change, but they increase the number of integration events and demand infrastructure that many legacy environments do not yet have.

**Benefits:**

- Review quality improves substantially, because reviewers can hold the entire change in mind. Large changes produce approvals rather than reviews, regardless of reviewer diligence.
- Defects are localized. When something breaks after a small change, the suspect set is small and the revert is safe, which is the single most effective antidote to fear of breaking changes in a legacy system.
- Merge conflicts and integration pain drop sharply, since changes are in flight for hours or days rather than weeks.
- Cycle time becomes predictable, because it stops being dominated by long queueing on a few large items.
- Progress becomes visible to stakeholders continuously, which reduces the pressure that itself drives teams toward big-bang delivery.

**Costs and Risks:**

- The overhead per change — pipeline runs, review requests, deployment steps — is paid more often. If the build and test cycle is slow, small batches make the slowness painful before they make anything better, so build times often have to be addressed first.
- Feature toggles accumulate. Without a discipline of removing toggles once a feature is fully released, the codebase acquires a new form of complexity and dead branches.
- Splitting work well is a genuine skill that takes time to develop. Early attempts often produce changes that are small but arbitrary, which are harder to review than one coherent larger change.
- In systems without automated tests, more frequent integration means more frequent opportunities to break production. Small batches and a basic safety net of tests need to be introduced together.

## How It Could Be

A team maintaining a logistics platform had a norm of one pull request per feature, which meant reviews of 1,500 to 4,000 changed lines arriving every two to three weeks. Reviews took four days on average, and the review comments were almost entirely superficial because no reviewer could reconstruct the intent of a change that size. The team adopted a thirty-minute reviewability guideline and started landing preparatory refactorings separately. The first feature done this way arrived as nine changes over eight days: four behavior-neutral extractions, three small additions behind a toggle, and two wiring changes. Average review turnaround fell to under four hours, and reviewers began raising substantive questions about error handling for the first time in memory.

A different team was blocked by a three-month branch for a payment provider migration that nobody dared to merge. They abandoned the branch and rebuilt the work incrementally: an adapter interface landed first with the old provider behind it and no behavior change, then the new provider implementation landed unused, then a toggle routed a single low-volume payment type to it. Each step was independently revertable, and the risky part of the migration was exercised in production against real traffic weeks before the cutover. The migration completed in six weeks rather than the projected three months, and the team retained the adapter as the seam for the next provider change.
