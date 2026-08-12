---
title: Definition of Ready
description: Agree what a piece of work must contain before the team starts it, so that half-specified work stops entering development and stalling there.
category:
- Requirements
- Process
- Team
problems:
- frequent-changes-to-requirements
- inadequate-requirements-gathering
- requirements-ambiguity
- poor-planning
- changing-project-scope
- scope-creep
- large-estimates-for-small-changes
- work-blocking
- eager-to-please-stakeholders
- incomplete-projects
- implementation-rework
- reduced-feature-quality
- constantly-shifting-deadlines
- delayed-project-timelines
- gold-plating
- missed-deadlines
- scope-change-resistance
- stakeholder-dissatisfaction
- time-pressure
- unrealistic-deadlines
- unrealistic-schedule
- feature-creep
- feature-factory
- large-feature-scope
- planning-dysfunction
- product-direction-chaos
layout: solution
---

## Description

A definition of ready is an agreed checklist that a piece of work must satisfy before the team commits to starting it: the problem is stated, the acceptance criteria are written, the affected systems are identified, the dependencies are known, and someone is available to answer questions. It is the entry gate that mirrors the definition of done at the exit. Its purpose is to stop the specific failure where work is pulled into development on the strength of a one-line description, stalls three days later on a question nobody can answer, and either sits blocked or proceeds on an assumption that turns out wrong. In legacy contexts one entry deserves particular weight: which existing behavior must not change. That question is answerable during preparation and enormously expensive to answer after the change has been built.

## How to Apply ◆

> In a legacy system the most consequential unknown is usually not what the new behavior should be, but what depends on the current behavior — and that is exactly what a rushed requirement omits.

- Write the checklist **with the team and whoever supplies the work**, not for them. A definition of ready imposed on a product owner becomes an obstacle to be routed around; one agreed with them becomes a shared standard.
- Keep it to **five to eight items**. A long checklist guarantees that nothing is ever ready, which leads to it being waived, which leaves the team where it started but with an extra ritual.
- Require **acceptance criteria expressed as observable behavior** — given this situation, when this happens, then this results. Criteria that cannot be checked cannot be finished, and work without a finishing condition is where scope creep enters.
- Include an item for **which existing behavior must be preserved**. This is the legacy-specific entry and it repays itself repeatedly: it forces someone to identify the affected consumers before the change is built rather than after it breaks them.
- Require that **dependencies and required access are identified** — the other team that must change something, the environment needed, the data required, the approval that will be needed. These are the items that turn into multi-day blocks once work has started.
- Name **who answers questions** for this item and confirm they are actually available during the period the work is planned. Work whose only informed stakeholder is on leave will stall regardless of how well it is specified.
- Require that the item is **small enough to complete within one cycle**. If it is not, splitting it is part of getting it ready, and the split usually reveals that some parts are ready and some are not.
- **Enforce it at the pull, not at the plan.** Work that does not meet the definition is not started; it goes back for preparation. Enforcement at planning alone lets items degrade in the interval.
- **Track how often items fail the check and why.** A consistent failure on the same item — acceptance criteria, or affected consumers — points at a specific gap in how work is prepared upstream, which is more useful than the checklist itself.

## Tradeoffs ⇄

> An entry gate prevents work from stalling mid-flight, at the cost of a queue in front of it and a real risk of being used to refuse work rather than to prepare it.

**Benefits:**

- Work stops stalling after it has started, which is the expensive kind of stall: context is loaded, capacity is committed, and the item occupies a slot while waiting.
- Requirement churn declines, because ambiguity is resolved before implementation rather than discovered during it.
- Estimates improve, since an item that meets the criteria is understood well enough to be estimated at all.
- The preservation question surfaces hidden consumers of existing behavior early, when accommodating them is a design choice rather than an emergency.
- Preparation work becomes visible as a real activity with real cost, rather than something expected to happen invisibly.

**Costs and Risks:**

- It creates a queue of unready work, and if nobody has capacity to prepare items, the team is blocked with the appearance of being well-organized.
- The checklist can become a weapon for refusing work, which damages the relationship with stakeholders and eventually gets the practice overridden from above.
- Over-specification is a real risk: pushed too far, a definition of ready becomes upfront analysis and destroys the ability to learn while building.
- Genuinely exploratory work — investigating a defect, spiking an unknown — cannot meet a criteria-based checklist and needs an explicit exemption, or the exemption will be taken informally for everything.
- Preparing items consumes the time of the people who are usually most in demand, and that cost has to be planned rather than assumed.

## How It Could Be

A team maintaining a public transport ticketing system found that roughly a third of started items stalled within the first three days, always for one of three reasons: nobody knew which downstream systems consumed the data being changed, the acceptance criteria were a single sentence, or the only person who understood the requirement was unavailable. They wrote a six-item definition of ready covering exactly those three plus dependencies, size, and a named question-answerer. In the first month, eleven of nineteen proposed items failed the check and went back. The product owner initially experienced this as obstruction. By the third month the failure rate was down to two in twenty, because items were being prepared differently, and the proportion of started work that stalled had fallen from a third to under five percent.

The preservation criterion produced the clearest single result. An item to change how discount codes were validated passed every other check easily. Answering "what existing behavior must not change" required someone to look, and the look found four batch jobs and one partner integration reading the validation results directly, two of which nobody on the team had known existed. The change was redesigned to keep the old output path intact and took an extra four days. The team's estimate of what the same change would have cost had it shipped without that knowledge — based on a comparable incident the previous year — was several weeks and a partner escalation.
