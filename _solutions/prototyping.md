---
title: Prototyping
description: Gather early feedback on functionality and usability
category:
- Process
- Requirements
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/prototyping
problems:
- assumption-based-development
- implementation-rework
- requirements-ambiguity
- poor-user-experience-ux-design
- misaligned-deliverables
- fear-of-change
- difficulty-quantifying-benefits
- rapid-prototyping-becoming-production
layout: solution
---

## How to Apply ◆

> Prototyping in legacy contexts focuses on reducing uncertainty about whether a proposed change or replacement will satisfy users who are accustomed to specific legacy workflows.

- Identify the riskiest aspects of the modernization — the features where legacy behavior is least understood or where the replacement design differs most — and prototype those first.
- Choose the appropriate fidelity level: paper sketches for workflow validation, clickable mockups for UI feedback, or working code prototypes for technical feasibility.
- Establish a clear "prototype boundary" with stakeholders: agree upfront that prototype code will be discarded and rewritten with proper engineering practices.
- Conduct structured feedback sessions where legacy system users compare prototype workflows with their current tasks, noting where the prototype improves, matches, or degrades their experience.
- Use prototypes to test integration approaches with legacy systems — for example, prototyping an API wrapper around a legacy database to validate data access patterns before committing to a full implementation.
- Track prototype feedback systematically and feed it into the product backlog as validated requirements.

## Tradeoffs ⇄

> Prototyping trades upfront time for reduced rework and improved requirements clarity, but requires discipline to prevent prototype code from becoming production code.

**Benefits:**

- Catches requirements misunderstandings and usability problems weeks or months before they would surface in a production implementation.
- Helps bridge the communication gap between developers who think in technical terms and users who think in workflows and business outcomes.
- Provides concrete evidence for modernization investment decisions rather than relying on theoretical arguments.
- Reduces resistance to change by letting users experience improvements firsthand rather than being told about them.

**Costs and Risks:**

- Prototype code that leaks into production is a common source of technical debt in modernization projects, especially when teams are under time pressure.
- Prototyping without clear goals can devolve into open-ended exploration that delays actual development.
- Users may form strong attachments to specific prototype designs, making it difficult to incorporate feedback from other user groups.
- The effort required for prototyping may be seen as wasteful by stakeholders who expect linear progress toward delivery.

## Examples

> The following scenario illustrates how prototyping guides decision-making in legacy modernization.

A manufacturing company needed to modernize its shop floor scheduling system, but operators were deeply skeptical that any replacement could handle the complex constraint-based scheduling they performed daily. The team built a working prototype that handled a simplified version of the scheduling problem and invited three experienced operators to test it with real production data. The operators quickly identified that the prototype's drag-and-drop interface was faster for routine scheduling changes but lacked the ability to express machine-specific constraints that the legacy system handled through obscure keyboard shortcuts. This feedback led to a hybrid interface design that combined modern UI patterns with a constraint expression panel, satisfying both usability goals and power-user requirements. The prototype sessions also converted the most skeptical operator into an advocate for the modernization effort.
