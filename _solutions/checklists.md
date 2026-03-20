---
title: Checklists
description: Systematically processing steps and requirements
category:
- Process
quality_tactics_url: https://qualitytactics.de/en/reliability/checklists
problems:
- inconsistent-quality
- quality-blind-spots
- poor-documentation
- inadequate-code-reviews
- complex-deployment-process
- rushed-approvals
- implementation-starts-without-design
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify repetitive, error-prone processes in the development lifecycle (deployments, code reviews, incident response) that would benefit from checklists
- Create concise checklists with clear, actionable items rather than vague recommendations
- Integrate checklists into existing workflows such as pull request templates, deployment scripts, or incident runbooks
- Review and update checklists regularly based on new findings, post-mortems, and changing requirements
- Keep checklists short enough to be practical (10-15 items maximum) while covering critical steps
- Distinguish between mandatory items that must be completed and optional items that are situational

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces errors of omission by making required steps explicit
- Ensures consistency across team members performing the same process
- Captures institutional knowledge in a form that survives team turnover
- Low-cost practice that can be adopted immediately without tooling changes

**Costs and Risks:**
- Checklists can become stale and lose relevance if not actively maintained
- Mechanical checkbox compliance without genuine engagement provides false confidence
- Overly detailed checklists slow down processes and encourage shortcuts
- Does not replace expertise and judgment for complex decisions

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy system team experienced recurring deployment failures because different team members performed deployments differently, each forgetting different steps. The team created a deployment checklist covering pre-deployment validation, backup verification, migration execution, smoke testing, and rollback criteria. The checklist was embedded in their deployment script as a series of confirmation prompts. Deployment failures dropped from an average of two per month to one per quarter. The checklist also became the starting point for automating deployment steps, with each item eventually being replaced by an automated check.
