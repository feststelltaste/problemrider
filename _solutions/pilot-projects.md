---
title: Pilot Projects
description: Prove a change on one small, real, reversible case before committing the organization to it — and let the result decide, including when the result is negative.
category:
- Process
- Culture
- Management
problems:
- resistance-to-change
- history-of-failed-changes
- past-negative-experiences
- fear-of-change
- inability-to-innovate
- premature-technology-introduction
- second-system-effect
- modernization-strategy-paralysis
- analysis-paralysis
- decision-paralysis
- cargo-culting
- reduced-innovation
- avoidance-behaviors
- delayed-decision-making
- fear-of-failure
- maintenance-paralysis
- modernization-roi-justification-failure
- perfectionist-culture
- rapid-prototyping-becoming-production
- short-term-focus
- process-software-misfit
layout: solution
---

## Description

A pilot project applies a proposed change — a technology, a practice, an architectural approach — to one small, real, reversible case, with agreed criteria for judging the result, before the organization commits to it broadly. Its function is to convert an argument into an experiment. Change proposals in legacy organizations tend to stall in a predictable pattern: the proponent argues from principle, the skeptics argue from the memory of the last initiative that was rolled out everywhere and failed, and neither position is falsifiable, so the decision is made on seniority or not at all. A pilot cuts through this by making the disagreement empirical and cheap to settle. It also directly addresses the specific damage caused by a history of failed changes, since the reasonable lesson from that history is not that change is bad but that unproven change applied everywhere is.

## How to Apply ◆

> Teams that have lived through two failed modernization programmes are not being irrational when they resist a third; a pilot is the form of proposal that takes their experience seriously.

- **Choose a case that is small, real, and representative.** Real matters most: a pilot on a toy problem proves nothing about a legacy environment, since the difficulty is always in the constraints the toy problem lacks.
- **Agree the success criteria before starting**, in writing, with the skeptics present. A pilot whose criteria are set afterward will be declared a success by its proponent and a failure by its opponents, which leaves everyone where they began.
- **Timebox it.** A pilot with no end date becomes a permanent parallel way of working, which is the worst outcome: the organization gets the cost of two approaches and the benefit of neither.
- **Make reversal genuinely possible and plan it explicitly.** If abandoning the pilot would be expensive, it is not a pilot — it is the first phase of a rollout that has not been decided.
- **Pick a team that wants to try it.** Volunteers make a fair test of the approach; conscripts make a test of their compliance, and a negative result from an unwilling team tells you nothing.
- **Record what happened honestly, including the parts that went badly.** A pilot report that reads as advocacy destroys the mechanism, because the next pilot will be assumed to be advocacy too.
- **Be willing to stop.** A pilot that cannot produce a negative result is a demonstration, not an experiment, and organizations learn very quickly to recognize the difference. The credibility of every subsequent pilot depends on at least some of them being abandoned.
- **Plan the second step before declaring victory.** What works for one willing team on one small case may not survive contact with a reluctant team on a large one, so extend gradually rather than announcing a general rollout.
- **Keep it visible.** A pilot conducted quietly persuades nobody. The point is partly to produce evidence and partly to let other teams watch, which is what makes voluntary adoption possible afterward.

## Tradeoffs ⇄

> Pilots make change decisions empirical and lower the risk of large failures, but they take time, and a small case may not predict what happens at scale.

**Benefits:**

- Disagreements about proposed changes become empirical questions with a cheap way to answer them, rather than contests of conviction.
- The organizational cost of a bad idea is bounded to one small case rather than a programme, which is the specific damage that a history of failed changes leaves behind.
- Skeptics get a legitimate route to be proven right, which paradoxically makes them far more willing to allow the attempt.
- Practical knowledge accumulates before the general rollout, so the broad version benefits from what the pilot learned rather than repeating it everywhere simultaneously.
- Evidence from a peer team is considerably more persuasive to other teams than any argument from management or from an external source.

**Costs and Risks:**

- Pilots take time, and where an organization faces a hard external deadline the sequential path may not fit.
- A small, willing team on a favorable case is not representative, and results genuinely may not scale — the pilot can produce false confidence as easily as justified confidence.
- Pilots that are never concluded leave the organization permanently running two approaches, with the overhead of both.
- The practice can be used to defer decisions indefinitely, with each pilot prompting a request for another before anything is committed.
- A pilot conducted by its proponent, with criteria set by its proponent, will succeed regardless of merit, and this pattern is quickly recognized and discounted.

## How It Could Be

An organization had attempted two large modernization programmes in eight years, both abandoned after substantial expenditure, and the engineering staff regarded any new architectural proposal with justified suspicion. A proposal to extract services from the monolith met the usual resistance. Rather than arguing, the architect proposed a pilot: one bounded capability — customer address validation — extracted by one volunteer team over eight weeks, with four criteria agreed in advance including a skeptic-proposed one that the monolith's deployment must not become more complicated. The pilot met three criteria and failed the fourth: the monolith's deployment did become more complicated, because the extracted service introduced a startup ordering dependency nobody had anticipated. That finding, produced in eight weeks for the cost of one small extraction, changed the approach for everything that followed — subsequent extractions were designed with a fallback path so that the monolith could start without them.

The willingness to stop was what eventually made the practice work in that organization. Their third pilot evaluated a workflow engine that a vendor had pitched convincingly and that a director favored. The agreed criteria included that a developer unfamiliar with the tool could modify a workflow within a day. Three developers tried; none succeeded within three days. The pilot was stopped and the finding written up plainly. The immediate effect was that a substantial licence purchase did not happen. The longer-term effect mattered more: the next pilot proposal was met with genuine engagement rather than suspicion, because the organization had demonstrated that a pilot could actually fail.
