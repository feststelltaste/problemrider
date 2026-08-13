---
title: Duplication Detection
description: "Systematically find where code has been copied, and check whether the\
  \ copies have drifted apart \u2014 because the dangerous duplicates are the ones\
  \ nobody knows about."
category:
- Code
- Testing
- Process
problems:
- code-duplication
- copy-paste-programming
- partial-bug-fixes
- regression-bugs
- inconsistent-execution
- high-technical-debt
- increased-bug-count
- maintenance-cost-increase
- difficult-to-understand-code
- brittle-codebase
- quality-degradation
- hidden-dependencies
- large-estimates-for-small-changes
- low-code-customization-sprawl
layout: solution
related_solutions:
- slug: code-hotspot-analysis
  similarity: 0.7
- slug: static-analysis-and-linting
  similarity: 0.7
- slug: code-metrics
  similarity: 0.65
- slug: code-reading-sessions
  similarity: 0.65
- slug: technical-debt-assessment
  similarity: 0.65
- slug: code-review-process-reform
  similarity: 0.65
---

## Description

Duplication detection systematically identifies passages of code that are substantially the same, by comparing normalized structure rather than raw text, so that renamed variables and reformatted whitespace do not hide a copy. Its value in legacy systems is not the total number it reports, which is close to useless as a quality measure. It is the discovery of specific copies that nobody remembered making. A defect fixed in one copy and not the others is one of the most common and most frustrating legacy failure modes: the bug reappears, is investigated as new, and is fixed again in a different copy, sometimes for years. Detection turns that from an unpleasant surprise into a checkable fact — when you change something, you can ask whether this logic exists elsewhere, and get an answer.

## How to Apply ◆

> The duplication that matters is not the passage everyone knows was copied; it is the one copied by someone who left in 2016, in a module nobody associates with this one.

- **Compare structure, not text.** Detection that works on raw characters misses everything where a variable was renamed or the formatting differs, which in practice is most of it. Normalizing identifiers and layout before comparing is what makes the results worth reading.
- **Recognize that clones come in degrees**: identical passages, passages differing only in names, passages restructured while preserving behavior, and passages that do the same thing written differently. Detection reliably finds the first three; the fourth is generally beyond it and needs human reading.
- **Ignore the headline percentage.** "This codebase is 14 percent duplicated" is a number without a decision attached. The useful output is a list of specific duplicate groups, and the total is mainly good for a trend.
- **Prioritize the copies that have drifted.** Two copies that are still identical are a maintenance cost; two copies that have diverged are a latent defect, because someone has already changed one and not the other. Diverged groups are the highest-value findings and should be reviewed first.
- **Prioritize the copies in code that is changing.** Duplication in a module nothing touches costs nothing. Crossing the detection results with change frequency reduces a list of hundreds to a handful worth acting on.
- **Look for duplication that crosses ownership boundaries.** Two teams maintaining the same logic independently is an organizational finding as much as a technical one, and it usually means a shared concept has never been named or given an owner.
- **Use it as a check before fixing a defect**, not only as a periodic report. "Does this logic exist anywhere else" is the question that prevents a partial fix, and it takes seconds to answer once detection is available.
- **Do not remove every duplicate.** Two passages that are similar today for unrelated reasons will diverge tomorrow, and merging them creates a coupling that is worse than the duplication. Deliberate duplication across separate business contexts is frequently the correct design.
- **Exclude what should be excluded**: generated code, vendored dependencies, and test fixtures where repetition aids readability. Detection that reports these will be ignored wholesale, including the findings that mattered.
- **Track the trend and pair it with a ratchet** so that new duplication does not accumulate while old duplication is being removed.

## Tradeoffs ⇄

> Detection finds copies nobody knew about and turns partial fixes into a preventable class of defect, but the raw output is noisy and removing duplication is not always an improvement.

**Benefits:**

- Unknown copies are found, which is the only way to stop the pattern where a defect is fixed in one place and recurs from another.
- Diverged copies surface as concrete findings, and divergence is direct evidence that a change has already been applied inconsistently.
- The check before a defect fix is cheap and prevents partial fixes, which is probably the largest practical benefit.
- Duplication crossing team boundaries reveals missing shared concepts and ownership gaps that no other analysis surfaces.
- The trend gives an objective signal about whether copy-paste practice is improving, which is otherwise a matter of impression.

**Costs and Risks:**

- The raw output is noisy and dominated by findings that do not matter, so it needs filtering by change frequency and divergence to be usable at all.
- It measures textual and structural similarity, not conceptual duplication, so it misses logic that was reimplemented rather than copied — often the more damaging kind.
- Treating the duplication percentage as a quality target invites removal of duplication that should have been left alone, producing premature abstractions that couple unrelated things.
- Merging duplicates creates coupling, and passages that are similar coincidentally will diverge later, at which point the shared abstraction becomes an obstacle.
- Configuring the exclusions and thresholds takes iteration, and an unconfigured run produces a report that discredits the practice.

## How It Could Be

A team had fixed the same rounding defect in an invoice calculation three times over two years, each time as a newly reported bug, each time in a different file. Running detection across their codebase found the calculation in five places, of which four had diverged from each other in small ways — one had the fix, two had different partial fixes, and one had never been touched since it was copied in 2014. That fifth copy was in a batch job producing a monthly report that a finance team reconciled by hand every month, a manual step whose origin nobody could explain. Consolidating the five into one implementation took a week, and the monthly reconciliation stopped.

The team's more lasting change was procedural rather than remedial. They added a single question to their defect-fixing routine: before fixing, check whether this logic appears elsewhere. Over the following year that check found duplicates in 11 of roughly 90 defect fixes, and in 4 of those cases the fix had to be applied in more than one place. Their earlier attempt to address duplication — a target percentage in the build — had been abandoned, because the number could only be improved by merging passages that were similar by coincidence, and two of those merges had later needed to be undone. The team's conclusion was that duplication detection was valuable as a lookup and worthless as a target.
