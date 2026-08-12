---
title: Mikado Method
description: Discover the true dependency graph of a large change by attempting it, recording what breaks, reverting, and fixing the prerequisites first.
category:
- Code
- Process
- Architecture
problems:
- maintenance-paralysis
- large-estimates-for-small-changes
- fear-of-change
- fear-of-breaking-changes
- second-system-effect
- increasing-brittleness
- strangler-fig-pattern-failures
- incomplete-projects
- monolithic-functions-and-classes
- history-of-failed-changes
- analysis-paralysis
- high-coupling-low-cohesion
- past-negative-experiences
- procrastination-on-complex-tasks
- long-lived-feature-branches
- refactoring-avoidance
layout: solution
---

## Description

The Mikado Method is a technique for making large structural changes to a codebase without ever leaving it broken. Instead of planning the change upfront — which requires knowledge of the dependency graph that nobody has — you attempt the change naively, observe precisely what breaks, revert immediately, and record the breakages as prerequisites. Each prerequisite is then attempted the same way, producing a tree of dependencies discovered empirically rather than guessed. Work proceeds from the leaves inward: every leaf is a change small enough to complete, verify, and commit on a working system. The method's central insight is that in a legacy codebase the dependency graph is unknowable in advance, so the compiler and the test suite should be used as instruments for discovering it. The characteristic failure it prevents is the multi-week branch that grows more broken with each attempted fix.

## How to Apply ◆

> This method exists specifically for the situation where a change appears simple, turns out to touch a dozen places, and the developer is three days in with nothing working and no way back.

- Write the **goal at the top of a sheet or file**, phrased as a concrete end state: "the ReportGenerator no longer reads the global configuration singleton." Vague goals produce trees that never terminate.
- **Attempt the goal directly and naively.** Make the change as if nothing else depended on it. Do not attempt to fix anything you break. The purpose of this step is measurement, not progress.
- **Record every error** the compiler, build, or test suite produces as a candidate prerequisite. Be specific: name the file and the reason, not "tests fail." The quality of the tree depends entirely on the precision of this recording.
- **Revert immediately and completely.** This is the discipline that makes the method work and the step people skip. The working tree returns to a known-good state after every experiment, so there is never a half-migrated system and never a reason to fear that the change cannot be abandoned.
- Attempt each **prerequisite the same way**, recursively. Prerequisites that can be completed without breaking anything are leaves; those that break things spawn their own prerequisites. The tree usually turns out to be deeper and narrower than expected, which is itself valuable information.
- **Complete and commit the leaves** one at a time, each on a working system with tests passing. Every leaf is independently valuable and independently revertable, so the work can be paused at any point without leaving debris.
- **Reattempt the original goal periodically** as leaves are completed. It often becomes achievable earlier than the tree suggests, because several prerequisites turn out to have shared a single underlying cause.
- Keep the tree **visible to the team** — a file in the repository, a shared diagram, a set of tickets. It communicates progress on work that otherwise looks like nothing is happening, and it lets someone else continue the effort, which matters for changes that span weeks.
- Use the **size and shape of the tree as a decision input**. A tree that grows to sixty nodes after two rounds is telling you something about the change's real cost, early enough to reconsider scope or approach rather than discovering it at week five.

## Tradeoffs ⇄

> The method trades apparent efficiency for genuine safety: repeatedly reverting feels wasteful, and is the reason the codebase never spends a day in a broken state.

**Benefits:**

- The system is working and committable at every point, so the change can be paused, handed over, or abandoned without loss — which removes most of the risk that makes large legacy changes frightening.
- The real dependency graph is discovered rather than estimated, which is why the method produces useful cost information for changes that upfront analysis systematically underestimates.
- Work is naturally decomposed into small, independently reviewable commits, without the developer having to design that decomposition in advance.
- Analysis paralysis is short-circuited: the way to find out what a change requires is to try it for twenty minutes, not to study the code for two days.
- Partially completed efforts leave the codebase better rather than worse, since every committed leaf is a genuine improvement even if the goal is never reached.

**Costs and Risks:**

- The repeated revert-and-retry cycle is genuinely slower per attempt and feels unproductive, particularly to developers under time pressure and to observers watching commit activity.
- It depends on fast feedback. If the build and test cycle takes forty minutes, the loop is too slow to be practical and build time must be addressed first.
- Without any automated tests, breakage cannot be detected reliably, so the method degrades to compiler-driven discovery only — useful in statically typed languages, weak in dynamically typed ones.
- The tree can grow large enough to be demoralizing, and teams sometimes abandon the effort at that point rather than reading the tree as the accurate cost estimate it is.
- Discipline is required to revert. A single "I'll just fix this one thing while I'm here" reintroduces the broken-branch failure mode the method exists to prevent.

## How It Could Be

A developer maintaining a shipping rate calculator was asked to make it testable so that a pricing change could be verified before release. The class read directly from a static configuration holder, a database singleton, and the system clock. Her first three attempts over two weeks had each ended with a branch too broken to finish and been abandoned. Using the Mikado Method, her first naive attempt — removing the static configuration reference — broke eleven compilation units in nine minutes, and she reverted. The tree that emerged had four prerequisites, one of which had three of its own. Over nine days she committed fourteen small changes, each with a green build, and the fourteenth made the original goal succeed on the first try. The pricing change that had prompted the work took an afternoon.

A team attempting to extract a customer module from a monolith used the method to decide against the extraction as scoped. Two rounds of naive attempts produced a tree with over fifty nodes, dominated by a shared database schema that eleven other modules read directly. Rather than proceeding, they used the tree as evidence in a planning discussion: the extraction was not a four-week task but a multi-quarter one, and its true first step was schema ownership, not code movement. They committed the eight leaves they had already identified as independently valuable — mostly interface extractions that improved testability — and redirected the remaining effort toward the schema. The tree became the planning document for the following two quarters.
