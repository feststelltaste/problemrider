---
title: Prototyping
description: Gather early feedback on functionality and usability
category:
- Process
- Requirements
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
related_solutions:
- slug: prototypes
  similarity: 0.95
- slug: wireframing
  similarity: 0.8
- slug: on-site-customer
  similarity: 0.75
- slug: user-stories
  similarity: 0.75
- slug: usability-tests
  similarity: 0.75
- slug: functional-spike
  similarity: 0.75
---

## Description

Prototyping is the practice of building low-commitment representations of a proposed change — ranging from paper sketches to clickable mockups to narrowly working code — specifically to reduce uncertainty about whether users accustomed to an existing legacy workflow will accept a proposed replacement, before that replacement is built in full. Unlike a finished increment of the system, a prototype's value lies entirely in the feedback it generates: fidelity is chosen deliberately for the question being asked, and an explicit agreement that the prototype's code will be discarded and rewritten is treated as part of the practice, not an afterthought. In legacy modernization specifically, prototyping is used to de-risk the two hardest sources of uncertainty in a replacement effort: whether the new design matches workflows users have never had to articulate because the legacy system simply does it that way, and whether a proposed integration approach against a legacy database or API will actually work before committing engineering time to building it in full. Structured feedback sessions that have legacy users directly compare the prototype against their current task, rather than evaluate it in the abstract, are what convert a subjective design opinion into concrete, actionable input for the backlog. The recurring failure mode is that prototype code, built under the same time pressure as the rest of the project, quietly becomes production code — a shortcut that reintroduces the very technical debt the modernization effort was meant to reduce, which is why establishing the prototype/production boundary up front is treated as inseparable from the practice itself.

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

## How It Could Be

> The following scenario illustrates how prototyping guides decision-making in legacy modernization.

A manufacturing company needed to modernize its shop floor scheduling system, but operators were deeply skeptical that any replacement could handle the complex constraint-based scheduling they performed daily. The team built a working prototype that handled a simplified version of the scheduling problem and invited three experienced operators to test it with real production data. The operators quickly identified that the prototype's drag-and-drop interface was faster for routine scheduling changes but lacked the ability to express machine-specific constraints that the legacy system handled through obscure keyboard shortcuts. This feedback led to a hybrid interface design that combined modern UI patterns with a constraint expression panel, satisfying both usability goals and power-user requirements. The prototype sessions also converted the most skeptical operator into an advocate for the modernization effort.
