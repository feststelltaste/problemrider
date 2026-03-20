---
title: Code Reviews
description: Systematic review of the source code by other developers
category:
- Process
- Code
- Team
quality_tactics_url: https://qualitytactics.de/en/maintainability/code-reviews/
problems:
- code-review-inefficiency
- inadequate-code-reviews
- insufficient-code-review
- large-pull-requests
- nitpicking-culture
- conflicting-reviewer-opinions
- review-bottlenecks
- style-arguments-in-code-reviews
- superficial-code-reviews
- review-process-avoidance
- review-process-breakdown
- reviewer-anxiety
- reviewer-inexperience
- perfectionist-review-culture
- extended-review-cycles
- reduced-review-participation
- team-members-not-engaged-in-review-process
- rushed-approvals
- inadequate-initial-reviews
layout: solution
---

## How to Apply ◆

> In legacy systems where quality controls are weak or absent, reforming the code review process is one of the most direct interventions for arresting further decay.

- Establish a mandatory pull request review policy even for maintenance changes; legacy systems often have no review gate at all, meaning every hotfix and workaround enters unchallenged.
- Rotate reviewers across unfamiliar modules deliberately — legacy codebases tend to have one person who "owns" each dark corner, and review rotation is the only way to spread that knowledge before they leave.
- Define a lightweight review checklist tailored to legacy risks: check for hidden side effects in shared global state, undocumented assumptions about external system behavior, and missing rollback paths for database changes.
- Keep change sets small by requiring that refactoring commits are separate from behavior changes; large mixed diffs in legacy code are nearly impossible to review meaningfully.
- Use asynchronous pull request review tooling (GitHub, GitLab, Bitbucket) even if the team is co-located — this creates a written audit trail of design decisions that compensates for the missing documentation typical in legacy systems.
- Automate all style and formatting checks via linting before code reaches human reviewers, so reviewers can focus on logic, coupling, and architectural intent rather than whitespace arguments.
- Nominate at least one reviewer per change who has no prior knowledge of the module — their confusion reveals undocumented assumptions that the original author has long stopped noticing.
- Acknowledge and record design decisions made during review as inline comments or linked decision logs; in legacy systems, this commentary often becomes the only available documentation for why code is structured the way it is.

## Tradeoffs ⇄

> Code review reform adds overhead to a development process that in legacy contexts is often already slow, so the trade-offs must be framed in terms of the cost of not reviewing.

**Benefits:**

- Stops the compounding of hidden workarounds by catching them before they are merged, reducing the rate at which the legacy codebase accumulates new debt.
- Spreads knowledge of notoriously siloed legacy modules across more team members, lowering the bus factor for systems where a single expert's departure could be catastrophic.
- Creates an incremental documentation layer in review comments and commit histories for code that has never been formally documented.
- Enforces coding standards consistently going forward, even when the existing code does not meet them, preventing further divergence of style across an already inconsistent codebase.
- Surfaces recurring patterns of fragility — if the same module keeps generating review findings, it signals where deeper refactoring investment is needed.

**Costs and Risks:**

- Reviewing legacy code is slower than reviewing greenfield code because reviewers must understand undocumented context before they can assess correctness; expect reviews to take longer than in a well-documented system.
- Review bottlenecks are especially damaging in legacy contexts where the team is small and often the same people are the only ones capable of reviewing specific areas.
- Without psychological safety, developers working on poorly written inherited code may feel that review comments are criticisms of their predecessors' work turned against them; review norms must explicitly address this.
- A superficial review culture — rubber-stamping approvals to meet process requirements — provides false confidence while adding calendar delay, which is worse than no process.
- Introducing mandatory review into a team accustomed to direct commits can create resistance; phasing in the requirement by module or risk level reduces friction.

## Examples

> The following scenarios illustrate how code review reform plays out in typical legacy modernization contexts.

A financial services firm inherited a monolithic Java application built over fifteen years by a succession of contractors. No review process existed; developers committed directly to the main branch. After three production incidents traced to unreviewed hotfixes cascading into each other, the team introduced pull request reviews with a two-person approval requirement for changes touching the payment processing module. Within two months, reviewers were flagging side effects in shared transaction state that the authors had not noticed — precisely the kind of issue that had caused the incidents. The written review history also became the team's first structured documentation of payment flow logic.

A government agency running a COBOL-based benefits system needed to bring on junior developers to supplement aging staff. Because only two engineers understood the core calculation engine, review was used as a structured knowledge transfer mechanism: every change to the calculation module required the junior developers to review it with guidance from the seniors, who explained domain rules in review comments rather than in meetings. Over six months, three additional team members could independently assess changes to the engine — measurably reducing the single-point-of-failure risk for a system that processed millions of claims per year.

A retail company's e-commerce platform had grown through acquisitions, combining three formerly independent codebases with three different coding conventions. The team introduced a review checklist that included a style convention check (which of the three standards was used) and a coupling check (did the change introduce dependencies across the formerly separate subsystems). These simple checklist items caught a disproportionate share of problems in the first quarter: not bugs in the narrow sense, but architectural violations that would have made future consolidation of the codebases significantly harder.
