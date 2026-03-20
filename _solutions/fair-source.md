---
title: Open Development Practices
description: Improve code quality through public code review, transparent issue tracking, and external contributions
category:
- Process
- Culture
quality_tactics_url: https://qualitytactics.de/en/maintainability/fair-source
problems:
- knowledge-silos
- insufficient-code-review
- poor-documentation
- limited-team-learning
- resistance-to-change
- feedback-isolation
- team-silos
layout: solution
---

## How to Apply ◆

> In legacy system contexts, open development practices increase transparency and attract fresh perspectives that can challenge entrenched assumptions about how the system must work.

- Make the codebase accessible to a broader audience within the organization (or externally for open source projects) by hosting it on platforms that support code review, issue tracking, and contributions.
- Establish contribution guidelines that make it clear how external contributors (from other teams or outside the organization) can report issues, suggest improvements, and submit changes.
- Use public issue tracking to make technical debt, known bugs, and improvement opportunities visible rather than hidden in private backlogs.
- Encourage cross-team code review by making pull requests visible and reviewable by anyone in the organization, not just the owning team.
- Document architectural decisions, coding conventions, and system constraints publicly so that potential contributors can onboard themselves.
- Create "good first issue" labels for legacy system cleanup tasks that external contributors can tackle without deep system knowledge.

## Tradeoffs ⇄

> Open development practices increase transparency and attract contributions but require governance and quality control for incoming changes.

**Benefits:**

- Brings fresh perspectives to legacy code that may benefit from outside viewpoints unencumbered by years of accumulated assumptions.
- Increases code review coverage by making code visible to a wider pool of reviewers.
- Improves documentation quality because public code must be understandable by people without institutional context.
- Reduces knowledge silos by making code, decisions, and discussions transparent to everyone.

**Costs and Risks:**

- Public visibility of legacy code quality may cause embarrassment or resistance from teams responsible for the code.
- External contributions require review effort and may not meet quality standards without clear contribution guidelines.
- Security-sensitive legacy code may not be appropriate for broad visibility.
- Maintaining open development infrastructure and responding to community contributions requires dedicated effort.

## Examples

> The following scenario illustrates how open development practices improve a legacy system.

A large enterprise with 20 development teams maintained a shared legacy framework that all teams depended on but only one team officially owned. By moving the framework to an internal open development model with public pull requests and issue tracking, the company enabled other teams to contribute fixes and improvements directly rather than waiting in the owning team's backlog. In the first year, 14 teams contributed 120 pull requests — 80% of which were bug fixes and documentation improvements that the owning team had never prioritized. The transparent issue tracker also revealed that three teams had independently built workarounds for the same framework limitation, leading to a coordinated fix that benefited everyone.
