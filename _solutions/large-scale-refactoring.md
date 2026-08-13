---
title: Large-Scale Refactoring
description: Roll one behavior-preserving change across many modules or repositories
  in tracked batches, with a named owner, so that sweeping refactorings actually finish.
category:
- Process
- Code
- Team
problems:
- technology-stack-fragmentation
- inconsistent-execution
- mixed-coding-styles
- code-duplication
- obsolete-technologies
- dependency-version-conflicts
- shared-dependencies
- high-technical-debt
- inconsistent-naming-conventions
- incomplete-projects
- organizational-structure-mismatch
- undefined-code-style-guidelines
- api-versioning-conflicts
- legacy-api-versioning-nightmare
- maintenance-paralysis
- over-reliance-on-utility-classes
- refactoring-avoidance
- strangler-fig-pattern-failures
- excessive-customization
- core-modification-of-standard-software
layout: solution
related_solutions:
- slug: small-change-batches
  similarity: 0.75
- slug: automated-code-migration
  similarity: 0.75
- slug: incremental-refactoring
  similarity: 0.75
- slug: mikado-method
  similarity: 0.75
- slug: preparatory-refactoring
  similarity: 0.7
- slug: code-review-process-reform
  similarity: 0.7
---

## Description

Large-scale refactoring is the organizational half of a sweeping behavior-preserving change: how a single transformation gets applied across dozens of modules or repositories owned by different teams, tracked to completion, and cleaned up afterwards. The tooling half — a recipe that performs the transformation — is usually the easier part. The reason such changes fail is almost never technical. It is that the change is applied to sixty percent of the estate, the remaining forty percent belongs to teams with their own priorities, nobody owns getting it finished, and the codebase is left permanently in two states. That end condition is worse than never having started, because now both the old and the new pattern must be supported indefinitely. The process exists to make finishing the default rather than the exception.

## How to Apply ◆

> A migration that stalls at eighty percent leaves the organization maintaining two idioms forever, which is the specific outcome the process is designed to prevent.

- **Give the change one named owner** who is responsible for it reaching completion, not for making the change. Sweeping changes with distributed ownership stall, reliably, and the stall is nobody's problem to notice.
- **Establish the true scope first.** Search the whole estate for the pattern before starting — including repositories nobody thinks of, generated code, and configuration. A migration whose scope is discovered progressively will be estimated wrongly and will find its worst cases last.
- **Pilot on one owned module** and measure what it actually took, including the review and the surprises. This gives a defensible estimate for the rest and, more importantly, produces a worked example to show other teams.
- **Prefer a compatibility shim over synchronized cutover.** Make the old and new forms coexist — a deprecated wrapper delegating to the new API — so each module can migrate independently. Requiring everything to switch at once puts the change at the mercy of the slowest team.
- **Batch the rollout** rather than opening sixty pull requests at once. Reviewers ignore a flood, and a batch of five or ten keeps the review meaningful and the merge conflicts manageable.
- **Track completion publicly** — a simple list of modules and their state. Visible progress is what sustains a change over the months it takes, and an unfinished list is what makes the remainder discussable rather than forgotten.
- **Make the change easy for the receiving team.** Provide the pull request, the recipe, the test results, and a one-paragraph explanation of why. A migration that asks other teams to do work will move at the speed of their priorities, which is slower than yours.
- **Prevent regression as you go.** Once a module is migrated, a lint rule, a ratchet, or a build check should stop the old pattern returning — otherwise the earliest migrations decay while the last ones are still in progress.
- **Delete the old path on a stated date**, and treat that as part of the change rather than as follow-up. Removing the shim is the step that gets dropped, and skipping it means the change never actually delivered its benefit.
- **Report the residue honestly.** Some modules will legitimately not migrate — frozen systems, third-party code, something scheduled for retirement. Naming them closes the change rather than leaving it perpetually at 94 percent.

## Tradeoffs ⇄

> A tracked process is what makes sweeping changes finish, at the cost of coordination overhead and a period during which two idioms coexist.

**Benefits:**

- The change actually completes, which is the difference between a consistent codebase and one that permanently carries two ways of doing the same thing.
- Teams are not blocked on each other, since the shim lets each migrate on its own schedule.
- The visible tracking sustains momentum across the months such changes take and makes the remaining work discussable instead of forgotten.
- Regression prevention applied as modules land means the early work does not decay while the late work is in progress.
- The pilot produces a real estimate and a worked example, which is far more persuasive to other teams than a request and a rationale.

**Costs and Risks:**

- Coordination across teams is real overhead, and it lands on the owner rather than being distributed.
- The compatibility shim is itself technical debt, and if the deletion step is skipped the organization has added a permanent layer rather than removed one.
- Large sweeping changes generate merge conflicts with everything in flight, which is a tax on every team during the rollout.
- Changes imposed on teams that see no benefit generate resentment, and their modules will be the ones still outstanding a year later.
- The process can be applied to changes that do not justify it, turning a nice-to-have consistency improvement into a months-long programme.

## How It Could Be

An organization with 40-odd services wanted to standardize on one HTTP client, having accumulated four across a decade. Two previous attempts had reached roughly half the services and stopped. The third attempt assigned one engineer as owner at 30 percent of their time. She searched the whole estate first and found 47 services rather than the 40 in the service registry, three of which had no identified owning team. She piloted on one service, measured it at about half a day including review, and wrote a recipe that handled the mechanical part. Rather than asking teams to migrate, she opened the pull requests herself in batches of six, each with the test results attached and a two-sentence explanation. Thirty-nine services were migrated in eleven weeks. The three unowned ones were escalated and became an ownership decision. Five were legitimately excluded — two frozen, three scheduled for retirement within the year — and were listed as such.

The deletion step was where the previous attempts had actually failed, in retrospect. Both had left compatibility wrappers in place, so the old clients remained in the dependency tree, kept receiving security patches, and continued to be used by new code because they were still available. The third attempt set a removal date at the outset and removed the three superseded clients on that date, which surfaced two services that had regressed to the old pattern during the rollout — caught precisely because someone was still looking. The organization's dependency count fell by three libraries, and the recurring security-patch work associated with them stopped.
