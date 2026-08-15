---
title: Checklists
description: Systematically processing steps and requirements
category:
- Process
problems:
- inconsistent-quality
- quality-blind-spots
- poor-documentation
- inadequate-code-reviews
- complex-deployment-process
- rushed-approvals
- implementation-starts-without-design
- inadequate-initial-reviews
- inconsistent-execution
- inconsistent-onboarding-experience
- review-process-breakdown
- reviewer-anxiety
- reviewer-inexperience
- unproductive-meetings
- code-review-inefficiency
- conflicting-reviewer-opinions
- insufficient-code-review
- superficial-code-reviews
- review-process-avoidance
layout: solution
related_solutions:
- slug: runbooks
  similarity: 0.8
- slug: portability-checklists
  similarity: 0.75
- slug: incident-management
  similarity: 0.75
- slug: ci-cd-pipeline
  similarity: 0.75
- slug: restore-points
  similarity: 0.7
- slug: blameless-postmortems
  similarity: 0.7
---

## Description

A checklist is a short, explicit, ordered list of the steps or requirements that must be completed for a given repetitive and error-prone process, used as an external memory aid that does not depend on any individual remembering every step correctly under pressure. It works by converting tacit expectations about "how this is supposed to be done" into a visible artifact that can be followed consistently regardless of who is performing the task or how experienced they are. In legacy systems, where deployment steps, review criteria, and incident procedures have often accumulated as unwritten conventions known only to a few long-tenured team members, checklists are a low-cost way to externalize that implicit knowledge before it is lost to turnover. They are particularly effective against errors of omission — the class of mistake where someone simply forgets a necessary step rather than performing a step incorrectly — which is exactly the failure mode that dominates in complex, rarely-changed legacy processes. Because checklists require no tooling investment to introduce, they are often the first, immediately actionable step in stabilizing a chaotic process, and can later serve as the specification from which automated checks are built one item at a time. Their value depends entirely on active maintenance, however: a checklist that is not updated as the underlying process changes quietly turns into a false source of confidence rather than a safeguard.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy system team experienced recurring deployment failures because different team members performed deployments differently, each forgetting different steps. The team created a deployment checklist covering pre-deployment validation, backup verification, migration execution, smoke testing, and rollback criteria. The checklist was embedded in their deployment script as a series of confirmation prompts. Deployment failures dropped from an average of two per month to one per quarter. The checklist also became the starting point for automating deployment steps, with each item eventually being replaced by an automated check.
