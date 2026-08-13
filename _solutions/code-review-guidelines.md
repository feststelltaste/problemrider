---
title: Code Review Guidelines
description: Agree in writing on what a review is for, what reviewers must check,
  what is merely an opinion, and when a change is good enough to merge.
category:
- Code
- Process
- Team
problems:
- superficial-code-reviews
- style-arguments-in-code-reviews
- nitpicking-culture
- perfectionist-review-culture
- conflicting-reviewer-opinions
- inadequate-initial-reviews
- rushed-approvals
- reviewer-anxiety
- review-process-breakdown
- code-review-inefficiency
- reviewer-inexperience
- bikeshedding
- team-members-not-engaged-in-review-process
- review-process-avoidance
- inconsistent-execution
- author-frustration
- clever-code
- cv-driven-development
- extended-cycle-times
- extended-review-cycles
- fear-of-conflict
- large-pull-requests
- mixed-coding-styles
- perfectionist-culture
- reduced-review-participation
- review-bottlenecks
- insufficient-code-review
- long-lived-feature-branches
- merge-conflicts
- poor-naming-conventions
- reduced-code-submission-frequency
- automated-tooling-ineffectiveness
- convenience-driven-development
- inadequate-code-reviews
- inconsistent-naming-conventions
- increased-risk-of-bugs
- inexperienced-developers
- undefined-code-style-guidelines
- low-code-customization-sprawl
layout: solution
related_solutions:
- slug: code-review-process-reform
  similarity: 0.8
- slug: code-conventions
  similarity: 0.75
- slug: lightweight-design-review
  similarity: 0.75
- slug: architecture-reviews
  similarity: 0.75
- slug: small-change-batches
  similarity: 0.7
- slug: code-reviews
  similarity: 0.7
---

## Description

Code review guidelines are a short, written agreement that answers the questions every review implicitly raises: what is this review supposed to catch, what must a reviewer look at, what counts as a blocking objection versus a suggestion, and when is a change good enough to merge. Most review dysfunction comes from these questions never being answered. Without a shared definition of "done reviewing," each reviewer substitutes their own: one waves changes through in two minutes, another blocks on variable naming, a third rewrites the design in the comments. In legacy systems the stakes are higher, because the reviewer often has no way to judge whether a change is safe except by reading it carefully — and no way to know how carefully is careful enough. Guidelines do not make reviews stricter or more lenient; they make them predictable, which is what turns review from a social gauntlet into an engineering control.

## How to Apply ◆

> In a legacy codebase the reviewer usually knows less about the touched module than the author does, so guidelines must tell reviewers what they can meaningfully check rather than assuming omniscience.

- Write down the **purpose of review** in one or two sentences and put it at the top of the document. A typical formulation for legacy work: "Reviews exist to catch defects, unsafe changes to fragile areas, and knowledge gaps — not to converge on one person's preferred style." Every later rule should be traceable to that purpose.
- Define a **checklist of what reviewers must check**: correctness of the change against its stated intent, error and edge-case handling, effects on callers of the changed code, test coverage for the new behavior, and any interaction with known-fragile modules. Keep it to five to seven items. A checklist that is too long gets skipped entirely, which is how superficial reviews start.
- Declare explicitly what reviewers must **not** spend review time on: formatting, import order, naming preferences that have no correctness impact, and alternative designs that are merely different rather than better. Move all mechanically checkable rules into automated linting and formatting so they never reach a human comment.
- Introduce a **comment taxonomy** so that every comment states its own weight. Three levels are usually enough: `blocking` (must be resolved before merge, and the reviewer states why it is unsafe or incorrect), `consider` (a suggestion the author may decline with a one-line reason), and `nit` (cosmetic, never blocks). This single convention resolves most conflicting-reviewer situations, because a disagreement between a `blocking` and a `nit` is no longer a standoff between equals.
- State the **tie-breaking rule** for genuine disagreements between reviewers: who decides, and within what time. A common rule is that the code owner of the affected module decides, and if there is no owner, the disagreement is escalated to a named technical lead within one working day rather than being argued in the pull request thread.
- Set an explicit **good-enough bar**: a change may merge when it is safe, tested, and better than what was there before — not when it is optimal. Write this down verbatim, because perfectionist review cultures are sustained by the unstated belief that approving imperfect code is a personal endorsement of it.
- Define **response-time expectations** in both directions: how quickly a reviewer is expected to pick up a review, and how quickly an author is expected to respond to comments. Without a stated expectation, review becomes the task everyone defers, and authors learn to route around it.
- Give **inexperienced reviewers an explicit mandate**. State that asking "I don't understand what this does" is a valid and valuable review comment, and that a reviewer is not expected to catch everything. Reviewer anxiety is usually the fear of missing something and being blamed for it; the guidelines should say plainly that review is a second pair of eyes, not a guarantee, and that responsibility for a defect is shared.
- Review the guidelines themselves in a retrospective every few months, using real examples of reviews that went badly. Guidelines that are never revisited stop matching how the team actually works and become another ignored document.

## Tradeoffs ⇄

> Written guidelines convert implicit, personally negotiated standards into an explicit team standard — which removes a great deal of friction, but also removes some of the flexibility that experienced reviewers use well.

**Benefits:**

- Reviews become predictable in scope and duration, which makes them easier to schedule and much harder to avoid or defer.
- Disagreements between reviewers get resolved by a stated rule instead of by seniority, persistence, or whoever is willing to keep arguing.
- Automating the mechanically checkable rules removes the large majority of low-value review comments, which is the fastest available fix for nitpicking and style arguments.
- New and less experienced reviewers can contribute immediately, because the checklist tells them what to look for instead of requiring them to already know.
- The explicit good-enough bar makes it socially acceptable to approve an imperfect change, which is a precondition for any team trying to improve a legacy codebase incrementally.

**Costs and Risks:**

- A checklist can become a substitute for thinking. Reviewers who tick items without engaging produce reviews that look thorough and catch nothing, so the checklist must be periodically tested against defects that reached production anyway.
- Guidelines that are imposed rather than agreed are ignored. The team must write them together, or at minimum ratify them, or the document has no authority when a disagreement actually occurs.
- Formalizing review can slow down trivial changes if the checklist is applied uniformly. A separate lightweight path for low-risk changes is usually needed, and defining "low-risk" in a legacy system is itself difficult.
- The comment taxonomy only works if senior reviewers use it honestly. A reviewer who labels every preference as `blocking` reintroduces the original problem with additional ceremony.

## How It Could Be

A team maintaining a 15-year-old insurance policy engine had review threads that routinely ran to forty comments, nearly all about naming and formatting, while two production defects in a quarter came from unhandled null cases that no reviewer had commented on. The team wrote a one-page guideline: a five-item checklist headed by "what breaks if this input is unexpected," a `blocking`/`consider`/`nit` prefix convention, and a rule that formatting was the formatter's job and no longer a valid comment. They enabled an auto-formatter on commit the same week. Within two months the average comment count per review fell from thirty-eight to nine, and the proportion of comments concerning error handling and caller impact rose from under five percent to roughly a third. The two subsequent quarters saw no production defects of the null-handling class.

A second team's problem was the opposite: reviews were approved within minutes and caught nothing, because reviewers felt unqualified to judge changes in modules they had never worked on. The guideline that changed this was a single sentence stating that "I cannot assess this safely" is a legitimate review outcome and obliges the team to find a second reviewer with the relevant knowledge, not the author to find a more permissive one. Reviewers stopped rubber-stamping, three chronically under-reviewed modules were identified by how often that outcome was invoked, and those modules became the first targets for deliberate knowledge sharing.
