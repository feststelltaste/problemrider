---
title: Debt Classification
description: Sort technical debt by whether it is actually charging you anything, so that effort goes to the debt that costs and the rest can be left alone without guilt.
category:
- Code
- Process
- Management
problems:
- high-technical-debt
- invisible-nature-of-technical-debt
- difficulty-quantifying-benefits
- maintenance-paralysis
- modernization-strategy-paralysis
- perfectionist-culture
- accumulation-of-workarounds
- increasing-brittleness
- brittle-codebase
- competing-priorities
- short-term-focus
- refactoring-avoidance
- analysis-paralysis
- increased-technical-shortcuts
- quality-compromises
- workaround-culture
- low-code-customization-sprawl
layout: solution
---

## Description

Debt classification sorts the known technical debt by whether it is actually costing anything, rather than by how unpleasant it is. The central distinction is between debt that charges interest and debt that is dormant: a poorly structured module that three people modify every week costs real money continuously, while an equally poor module that nothing has touched in four years costs nothing at all and will cost nothing until someone touches it. Teams do not naturally make this distinction, because the emotional response to bad code is driven by how it reads rather than by what it costs. The result is effort spent on the code that is most offensive rather than the code that is most expensive, and a pervasive sense that the whole system is a liability. Classification is what makes the debt proportionate — and most of it, on inspection, turns out not to matter.

## How to Apply ◆

> A legacy system contains an enormous amount of debt that will never be paid because nothing will ever go near it, and identifying that portion is as valuable as identifying the rest.

- **Establish whether each item is interest-bearing.** The test is empirical, not aesthetic: has this code been changed in the last year, is it involved in incidents, does it slow down work that people actually do? Change frequency from version control answers most of it in an afternoon.
- **Separate deliberate from accidental debt.** Debt taken knowingly under time pressure, with a reason, is a different management problem from debt that accumulated because nobody knew better. The first needs a repayment decision; the second needs a capability intervention, and treating them alike addresses neither.
- **Distinguish debt that blocks from debt that slows.** Something that makes a change impossible, or makes an entire class of work unreachable, ranks above something that makes every change fifteen percent more tedious — even when the second is more widespread and more irritating.
- **Mark the dormant debt explicitly as accepted**, in writing, rather than leaving it on a backlog. An item that is knowingly not going to be addressed should say so and say why. This is the step that shrinks the list to something a team can look at without despair.
- **Re-classify when circumstances change.** Dormant debt becomes interest-bearing the moment a roadmap item touches that area, so the classification should be revisited when plans change rather than annually.
- **Use the classification to set the response**, not just the order. Interest-bearing and blocking debt gets remediated; interest-bearing and slowing debt gets addressed opportunistically through preparatory refactoring; dormant debt gets contained behind an interface or left alone; and debt in code scheduled for deletion gets nothing.
- **Record the reasoning per item**, briefly. Classification without reasons becomes contested every time someone new looks at the list, and the reasons are what let a successor re-classify sensibly.
- **Report the profile, not just the total.** "We have 140 debt items" is frightening and useless. "Of 140 items, 22 are interest-bearing, 6 of those are blocking, and 118 are dormant and accepted" is a management statement.
- **Watch for the aesthetic reflex.** The strongest advocacy usually attaches to the debt that is most unpleasant to read, and that correlation with actual cost is weak. Requiring evidence for the interest-bearing classification is what keeps this honest.

## Tradeoffs ⇄

> Classification directs effort at the debt that costs and makes the rest explicitly acceptable, but it requires judgement calls that will sometimes be wrong and can be used to dismiss real problems.

**Benefits:**

- Effort concentrates on debt that actually costs, which is typically a small fraction of what a team perceives as debt.
- The list becomes bounded and reviewable, because the dormant majority is explicitly accepted rather than sitting as permanent unfinished business.
- The dread becomes proportionate. Much of the anxiety about a legacy system comes from treating all of its flaws as equally live, and they are not.
- Different debt types get different responses, which is more efficient than treating everything as a candidate for refactoring.
- The profile is communicable to management in a way a raw count is not, which makes remediation requests credible.

**Costs and Risks:**

- Dormant debt occasionally becomes urgent without warning — a security advisory, an unexpected feature request — and code accepted as untouchable can turn out to need touching.
- The classification requires judgement, and a team under pressure will classify inconveniently expensive items as dormant.
- Explicitly accepting debt can be read as tolerating poor quality, and it needs framing as a prioritization decision rather than a standard.
- Change frequency is a proxy. Code that is avoided precisely because it is frightening looks dormant in the data while being a serious liability.
- Re-classification is easy to skip, leaving a stale classification that says something is dormant when the roadmap has just aimed at it.

## How It Could Be

A team maintaining a manufacturing system had a technical debt backlog with 187 items, accumulated over six years, which nobody looked at because looking at it was demoralizing. They classified it over three days using change frequency from version control and their incident record. Twenty-nine items were interest-bearing. Five of those were blocking — they made specific planned work impossible rather than merely harder. The remaining 158 were dormant, and 41 of those were in code that had not been modified since 2018. The 158 were marked accepted with a one-line reason each and moved out of the active list. The remaining 29-item list was, for the first time, something the team looked at in planning. Three of the five blocking items were addressed in the following quarter.

The deliberate-versus-accidental split changed a management conversation. Classifying the 29 interest-bearing items by origin showed that 19 were deliberate shortcuts taken under specific deadlines, all of which had been flagged at the time and none of which had ever been revisited. That was not a code quality problem — it was a process problem, and the fix was a rule that any shortcut taken under deadline pressure carried a mandatory review at the next quarter. The other 10 were accidental, concentrated in work done by two developers during a period when the team had no review practice worth the name. That was a capability problem, and it was addressed by coaching rather than by refactoring. Neither response would have been chosen from a list that treated all 29 items as the same kind of thing.
