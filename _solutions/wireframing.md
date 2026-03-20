---
title: Wireframing
description: Create preliminary visual representations as a basis for discussion
category:
- Requirements
- Process
quality_tactics_url: https://qualitytactics.de/en/usability/wireframing/
problems:
- poor-user-experience-ux-design
- implementation-starts-without-design
- requirements-ambiguity
- stakeholder-developer-communication-gap
- misaligned-deliverables
- implementation-rework
- feature-gaps
- user-frustration
layout: solution
---

## How to Apply ◆

> Legacy system modernization often begins coding before the team has a shared understanding of what the improved interface should look like. Wireframing creates low-cost visual representations that align stakeholders before development begins.

- Create low-fidelity wireframes using simple tools like pen and paper, Balsamiq, or Figma before any development work starts on interface changes. The goal is to explore layout and interaction options, not to produce polished designs.
- Use wireframes to facilitate discussions with stakeholders and users about what information and actions should appear on each screen, how they should be organized, and what the workflow between screens should be.
- Create wireframes for both the current state and the proposed future state so stakeholders can see what will change and provide informed feedback.
- Test wireframes with representative users through paper prototype testing or clickable prototype walkthroughs to identify usability issues before any code is written.
- Iterate on wireframes rapidly based on feedback. The cost of changing a wireframe is negligible compared to the cost of changing implemented code.
- Use wireframes to document interface decisions and serve as a specification for developers, reducing ambiguity about what should be built.

## Tradeoffs ⇄

> Wireframing prevents costly rework by validating design decisions early, but adds a design step that teams accustomed to coding directly may resist.

**Benefits:**

- Catches design problems and requirements misunderstandings before code is written, when changes are cheapest and easiest.
- Creates a shared visual language between developers, stakeholders, and users, reducing the communication gap that leads to misaligned deliverables.
- Reduces implementation rework caused by building the wrong thing because the team did not have a clear picture of the target.
- Enables rapid exploration of multiple design alternatives at low cost before committing to one approach.

**Costs and Risks:**

- Adding a wireframing step extends the timeline before development begins, which can be perceived as delay in organizations that prioritize speed.
- Wireframes that are too polished can set unrealistic expectations about the final visual quality, especially if the legacy technology stack limits what is achievable.
- Stakeholders may focus on wireframe aesthetics rather than structure and interaction flow, derailing discussions about what matters.
- Wireframes can become outdated quickly if not maintained as the design evolves during implementation, becoming misleading artifacts.

## Examples

> Many legacy modernization projects fail because the team builds an interface that nobody reviewed or validated before implementation.

A legacy billing system is being modernized and the development team plans to rebuild the invoice creation screen. Without wireframes, the team builds the new screen based on their understanding of the requirements, spending three sprints on implementation. When stakeholders review the result, they discover that the workflow does not match the accounting team's process: the team built a single-step form when the accountants need a multi-step workflow with approval checkpoints. The entire screen must be substantially reworked. For the next module, the team adopts wireframing. They spend two days creating wireframes of the proposed payment reconciliation screen, review them with the accounting team, and iterate through three versions based on feedback. When development begins, the team has a validated design that matches the actual workflow. The implementation proceeds smoothly with no rework, and the accounting team feels ownership of the design because their input shaped it from the start.
