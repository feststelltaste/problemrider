---
title: Preparatory Refactoring
description: Before making a change, restructure the code so that the change becomes easy — then make the easy change, as two separate steps.
category:
- Code
- Process
problems:
- refactoring-avoidance
- large-estimates-for-small-changes
- feature-creep-without-refactoring
- complex-implementation-paths
- defensive-coding-practices
- accumulation-of-workarounds
- increasing-brittleness
- copy-paste-programming
- maintenance-paralysis
- increased-technical-shortcuts
- difficult-to-understand-code
- workaround-culture
- bloated-class
- global-state-and-side-effects
- god-object-anti-pattern
- long-lived-feature-branches
- merge-conflicts
- convenience-driven-development
- monolithic-functions-and-classes
- over-reliance-on-utility-classes
- poor-encapsulation
- procrastination-on-complex-tasks
- tangled-cross-cutting-concerns
layout: solution
---

## Description

Preparatory refactoring means that when a change is hard to make, you first restructure the code so that the change becomes easy, and only then make it — as two separate, separately verified steps. It resolves the recurring dilemma of legacy maintenance: the code is not shaped for the change being asked for, so the developer either forces the change into the existing shape, adding another workaround, or embarks on an open-ended cleanup that is hard to justify and hard to review. The discipline avoids both. The refactoring is scoped by a specific purpose — make this particular change easy — which bounds it, and it is behavior-preserving, which makes it verifiable by existing tests. The subsequent functional change is then small enough to review properly. The practice also converts improvement from a separate activity that must be funded into a normal part of doing the work.

## How to Apply ◆

> Legacy code is shaped by the changes that were made to it, which is why the next change never fits: the shape reflects requirements from years ago that nobody has revisited since.

- **Attempt the change first, briefly.** The difficulty tells you what restructuring is needed. Refactoring speculatively, before knowing what the change requires, produces a different shape that the change still does not fit.
- **Revert that attempt** and do the refactoring on its own. Mixing the two is the failure mode: a diff containing both a move and a behavior change cannot be reviewed, because the reviewer cannot tell which lines are supposed to be behavior-neutral.
- Keep the refactoring **scoped to what the change needs.** "Extract this method so the new branch has somewhere to live" is bounded and defensible; "clean up this class" is neither, and it is what makes managers distrust refactoring.
- **Commit and ideally ship the refactoring separately.** It is behavior-preserving, so it carries low risk and can go to production on its own. If the functional change is then delayed or cancelled, the codebase still improved.
- **Verify behavior preservation** with whatever safety net exists — the existing tests, or characterization tests written for the occasion. Where none exists, use conservative dependency-breaking transformations small enough to verify by reading.
- Include the preparatory step **in the estimate rather than hiding it.** Presenting it as part of the work is honest and makes the real cost of the code's current shape visible. Concealing it as padding erodes trust when it is discovered.
- Recognize the **signal that the code is telling you something**. A change that requires touching six places is telling you those six places share a concept that has never been named. The refactoring that fixes that is the valuable one.
- Know when **not to prepare**: code that is scheduled for deletion, code that has not changed in years and is not changing now, and genuine emergencies. Preparatory refactoring earns its cost where change is ongoing.
- If the preparation turns out to be **much larger than the change**, stop and treat it as its own piece of work with its own decision. Discovering that is a useful result, and continuing anyway is how a two-day task becomes a three-week branch.

## Tradeoffs ⇄

> Splitting the work makes both halves reviewable and makes improvement routine, at the price of a longer path to the functional change and a discipline that erodes under pressure.

**Benefits:**

- Both steps become individually reviewable, which is a substantial quality improvement over a single diff that mixes movement with behavior change.
- The functional change ends up small, obvious, and safe, which is where most of the defect reduction comes from.
- Improvement happens continuously, in the areas that are actually being changed, without needing separate funding or a dedicated initiative.
- The refactoring is low-risk and independently shippable, so partially completed work still leaves the codebase better.
- It counteracts the accumulation of workarounds directly, because the alternative — forcing the change into a shape that does not fit — is precisely how workarounds accumulate.

**Costs and Risks:**

- The functional change takes longer to arrive, and under deadline pressure the preparation is the step that gets skipped, which is when it was most needed.
- Without tests, behavior preservation cannot be verified, so the practice depends on either having a safety net or restricting itself to transformations verifiable by reading.
- Scope discipline is genuinely hard. Preparatory refactoring drifts into general cleanup easily, and once it does, it becomes the unbounded activity that management learns to refuse.
- Preparation targeted at the wrong change wastes effort and can leave the code shaped for a requirement that never arrives.
- Two commits where there was one adds process overhead, which is real when the build and review cycle is slow.

## How It Could Be

A developer was asked to add a second discount type to a checkout flow. The existing discount logic was a 200-line method with the single discount type woven through six conditional branches. Her first attempt at adding the new type directly produced a method she could not follow after twenty minutes. She reverted, spent a day extracting the discount calculation into a small interface with the existing logic as its only implementation, verified against the existing tests, and shipped that as its own change. The next day the new discount type was a new implementation of that interface: 40 lines, with its own tests, and a two-line change at the call site. Total elapsed time was slightly more than a forced-in change would have taken. When a third discount type was requested four months later, it took an afternoon.

A team used the same discipline to make a modernization argument. Their tracking showed that a third of their preparatory refactorings turned out to be larger than the change that prompted them, and that these clustered in two subsystems. Rather than continuing to absorb the cost invisibly inside individual tasks, they recorded it: over one quarter, 31 developer-days of preparation had been spent in those two subsystems for changes whose functional content was 6 days. The ratio, presented as a measured figure rather than as a complaint, was what secured a dedicated effort on the worse of the two — and it was derived entirely from work the team was doing anyway.
