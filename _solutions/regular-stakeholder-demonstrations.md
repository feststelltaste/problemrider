---
title: Regular Stakeholder Demonstrations
description: Show working software to the people who asked for it on a fixed cadence, so that misunderstandings surface in days rather than at delivery.
category:
- Communication
- Business
- Process
problems:
- stakeholder-confidence-loss
- stakeholder-dissatisfaction
- stakeholder-frustration
- feedback-isolation
- eager-to-please-stakeholders
- feature-factory
- product-direction-chaos
- planning-credibility-issues
- reduced-feature-quality
- inadequate-requirements-gathering
- requirements-ambiguity
- delayed-value-delivery
- missed-deadlines
- cascade-delays
- changing-project-scope
- constantly-shifting-deadlines
- deadline-pressure
- delayed-project-timelines
- gold-plating
- incomplete-projects
- poor-communication
- poor-planning
- scope-change-resistance
- unrealistic-deadlines
- unrealistic-schedule
- communication-risk-outside-project
- declining-business-metrics
- feature-creep
- frequent-changes-to-requirements
- market-pressure
- poor-project-control
- unclear-goals-and-priorities
- unproductive-meetings
- process-software-misfit
layout: solution
---

## Description

A regular demonstration is a short, fixed-cadence session in which the team shows working software — not slides, not status, not percentages complete — to the people who requested it and who will use it. Its function is to convert the abstract question "is this what you meant" into a concrete one that can be answered by looking. Written requirements are always incomplete, and the gap between what a stakeholder asked for and what they meant is not discoverable by asking more carefully; it is discoverable by showing them something. The cadence matters as much as the content, because a demonstration every two weeks bounds the possible misunderstanding to two weeks of work. In legacy modernization the practice serves a second purpose: it is the only reliable way to show progress on work whose visible output is otherwise zero for months.

## How to Apply ◆

> Legacy work often produces nothing a stakeholder can see for long stretches, which is precisely the condition under which confidence erodes and pressure builds.

- Demonstrate **running software in an environment that resembles production**, using realistic data. A walkthrough of a design or a description of what was built does not surface the misunderstandings that seeing the actual behavior does.
- Hold it on a **fixed schedule** whether or not there is much to show. Cancelling because progress was thin removes the feedback exactly when the team most needs to check that it is still on course, and it teaches stakeholders that the meeting signals good news only.
- Invite **the people who will actually use the system**, not only their managers. The gap between what a department head describes and what the person doing the work needs is where a large share of unusable features originate.
- Have the **person who built it demonstrate it**. Questions get answered directly, and the developer hears the reaction unfiltered, which changes subsequent decisions more than any relayed summary.
- **Show the unfinished and the imperfect** deliberately. A demonstration that only shows polished work delays feedback until changing course is expensive, and it trains stakeholders to expect a finished appearance that then constrains what the team is willing to show.
- For work with **no visible surface** — a migration, a performance effort, a dependency upgrade — demonstrate the evidence instead: the before-and-after measurement, the parallel-run comparison, the traffic now served by the new path. The point is that something verifiable is shown, not that it is a screen.
- **Record decisions and requested changes** during the session and feed them into the ranked backlog rather than into the current work. A demonstration that silently expands scope is how a well-intentioned team becomes unable to finish anything.
- Keep it **short and unrehearsed**. A session that takes two days to prepare will be prepared less often, and preparation effort tends to go into making things look finished rather than into getting them looked at.
- Use the cadence to **build the credibility that planning discussions need**. Stakeholders who have seen working software every two weeks for six months respond very differently to a forecast than those who have seen status reports.

## Tradeoffs ⇄

> Frequent demonstrations catch misunderstandings early and rebuild trust, but they consume time from busy people and expose work before it is comfortable to show.

**Benefits:**

- Misunderstandings are caught within one cadence period instead of at delivery, which is the difference between an adjustment and a rewrite.
- Stakeholder confidence recovers, because progress is observed rather than asserted — this is usually far more effective than any improvement in reporting.
- The team hears reactions directly, which improves subsequent design decisions in ways that written feedback does not.
- Requests are captured in one place at a predictable time, which reduces the ad-hoc side-channel requests that otherwise arrive continuously and disrupt planning.
- Invisible work becomes visible when the evidence is demonstrated, which protects modernization efforts from being cancelled for apparent lack of progress.

**Costs and Risks:**

- It consumes the time of senior stakeholders on a recurring basis, and attendance decays unless the sessions are consistently worth attending.
- Demonstrations invite requests, and without the discipline of routing them to the backlog they become a mechanism for continuous scope expansion.
- Preparing a demonstrable state can distort priorities toward what looks good rather than what matters, particularly if the audience reacts mainly to surface appearance.
- Showing unfinished work requires enough trust that roughness is not read as incompetence, and in a low-trust relationship the early sessions can make things worse before better.
- Some legacy work genuinely has nothing to show for weeks, and forcing a demonstration in those periods produces contrived content that undermines the credibility of the format.

## How It Could Be

A team rebuilding the pricing engine of an e-commerce platform worked for five months against a written specification and demonstrated the result at the end. Roughly forty percent of it had to be reworked: the specification's description of promotional stacking matched the document its author had written, but not how the pricing analysts actually worked, which involved manual overrides that had never been written down anywhere. On the next major effort the team demonstrated every two weeks to two analysts and one manager. The equivalent misunderstanding — this time about how tax exemptions interacted with bulk discounts — surfaced in the third session, eleven days into the work, and cost an afternoon to correct.

A modernization team faced the opposite problem: eight months of a database migration with nothing visible to show, and growing pressure to justify the investment. They began demonstrating evidence rather than screens. Each session showed the parallel-run comparison dashboard: how many record types were now being written to both systems, how many discrepancies had been found that week, and which had been resolved. The steering group's questions shifted from "when will this be finished" to substantive discussion of the discrepancy categories, and the effort was extended twice without contention. Two of the discrepancy patterns shown in those sessions turned out to be pre-existing data quality problems that the business side had been working around manually for years, and which nobody had previously been able to quantify.
