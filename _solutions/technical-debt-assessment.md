---
title: Technical Debt Assessment
description: Investigate one area in depth, on a timebox, and produce a written picture of what is actually wrong there — replacing a general dread with specific findings.
category:
- Code
- Architecture
- Process
problems:
- high-technical-debt
- invisible-nature-of-technical-debt
- brittle-codebase
- increasing-brittleness
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- maintenance-paralysis
- modernization-strategy-paralysis
- fear-of-change
- legacy-system-documentation-archaeology
- maintenance-bottlenecks
- difficult-to-understand-code
- analysis-paralysis
- accumulation-of-workarounds
- large-estimates-for-small-changes
- maintenance-cost-increase
- refactoring-avoidance
- workaround-culture
layout: solution
---

## Description

A technical debt assessment is a timeboxed, structured investigation of one bounded area of a system, ending in a written document that states what is wrong there, how bad each thing is, and what it would take to address. It differs from a metrics dashboard, which reports numbers without judgement, and from an architecture review, which is periodic and broad. This is deliberately narrow and deep: one subsystem, one to three weeks, one document. Its purpose is to convert a general condition — "that module is a nightmare" — into a finite list of named findings. That conversion is the point. Teams and managers experience legacy debt as an unbounded dread, and an unbounded problem cannot be planned, funded, or prioritized. A list of eleven specific findings, each with a size, is a problem that can be worked on, even when the list is alarming.

## How to Apply ◆

> The area everyone avoids talking about specifically is usually the one where an assessment returns the most, precisely because nobody has looked.

- **Pick one bounded area**, not the system. A subsystem, a module, a data flow. Assessing everything produces a document nobody reads and findings too general to act on. Hotspot data — change frequency crossed with defect involvement — picks the area when intuition is uncertain.
- **Timebox it hard**, one to three weeks depending on size, and state up front that the output is a picture rather than a complete inventory. Assessments without a fixed end grow until they are abandoned.
- **Use several lenses and record which found what**: reading the code, the change history, the incident record, the test coverage, the dependency structure, and interviews with whoever works there. Each surfaces things the others miss, and the change history in particular reveals problems no code reading finds.
- **State each finding concretely**: what it is, where it is, what it costs today, what happens if nothing is done, and a rough size to address. A finding without a cost and a size is an observation, and observations do not get funded.
- **Separate what hurts from what is merely ugly.** Most of what an assessment could report is aesthetic and costs nothing. Reporting it dilutes the findings that matter and is why assessments acquire a reputation for producing wish lists.
- **Include what is working.** An assessment that finds only problems reads as an indictment of the people who built it, which makes the next one harder to arrange and makes the team defensive rather than collaborative.
- **Have someone from outside the area do it, with someone from inside.** The outsider asks the questions that familiarity has suppressed; the insider prevents the outsider from misreading deliberate decisions as mistakes.
- **Write it for two audiences.** A one-page summary in cost and risk terms, and the detailed findings beneath. An assessment readable only by engineers cannot do the job of making the debt graspable to the people who fund the work.
- **End with a recommended sequence**, not just a list. Which three findings first, and why. A list of eleven findings with no ordering hands the prioritization problem back to whoever commissioned the assessment.
- **Re-assess after remediation** to check the findings actually closed. Assessments that are never revisited become historical documents that describe a system that no longer exists.

## Tradeoffs ⇄

> A deep assessment converts vague dread into a finite, plannable list, at the cost of real effort and the risk of producing a document that is read once and shelved.

**Benefits:**

- The problem becomes bounded. A named list of findings can be prioritized, sized, and funded; a general condition cannot, and this is usually the actual blocker.
- Fear becomes proportionate. Assessments routinely find that a feared subsystem has three real problems and a great deal of merely unpleasant code, which changes how the team approaches it.
- Findings carry costs and sizes, which is what allows debt work to enter a prioritization discussion on the same terms as everything else.
- The written record survives staff changes, so the understanding is not lost when the person who investigated moves on.
- The multi-lens approach surfaces problems no single tool detects, particularly the ones visible only in the change history or the incident record.

**Costs and Risks:**

- One to three weeks of capable people produces no working software, and that cost is felt immediately.
- Assessments frequently end as documents nobody acts on, which wastes the effort and teaches the organization that assessment is theatre.
- Findings can read as criticism of the people who wrote the code, and in a blame culture the assessment will be resisted or quietly neutered.
- A bounded scope means everything outside it is unassessed, and the worst problem may be in the area that was not chosen.
- The picture goes stale. In an actively developed area an assessment is a snapshot with a shelf life of months.

## How It Could Be

A team described their billing subsystem as "the part where things go wrong" and had for two years been unable to get any improvement funded, because every proposal amounted to asking for time to make a bad thing better. They assessed it over two weeks, two people, four lenses. The result was eleven findings, of which four were rated as costing something today: a duplicated tax calculation that had drifted between its two copies, a retry loop that silently swallowed a specific failure class, an absence of tests around the proration logic, and a scheduled job whose failure was not alerted. Sizes ranged from two days to six weeks. Seven further findings were recorded as ugly but harmless. The four costly findings were funded within a month — not because the subsystem had become less bad, but because the request was now for a specific six-week piece of work rather than for an open-ended improvement.

The proportionality effect surprised the team more than the funding did. Their collective sense had been that the subsystem was uniformly dangerous and that touching any part of it was risky. The assessment found that roughly 70 percent of it was tedious but straightforward, and that the risk was concentrated in two files. Developers who had been routing around the whole subsystem began working in the safe majority of it normally. The one-page summary — four findings, their monthly cost, and a recommended order — was also the first document about that subsystem that the finance director had ever read, and it was what changed the conversation from "engineering wants to rewrite things" into a discussion about sequencing.
