---
title: Code Hotspot Analysis
description: Combine change frequency from version control with complexity and defect data to identify the small share of code where improvement effort actually pays off.
category:
- Code
- Process
- Management
problems:
- maintenance-bottlenecks
- bloated-class
- excessive-class-size
- copy-paste-programming
- increasing-brittleness
- increased-bug-count
- maintenance-cost-increase
- high-technical-debt
- invisible-nature-of-technical-debt
- monolithic-functions-and-classes
- delayed-issue-resolution
- automated-tooling-ineffectiveness
- feature-creep-without-refactoring
- system-stagnation
layout: solution
---

## Description

Code hotspot analysis identifies where improvement effort will return the most by crossing two data sources that most teams already have: how often each file changes, taken from version control history, and how complex or defect-prone each file is. Complexity alone is a poor guide — a complicated module that nobody has modified in five years costs nothing to leave alone. Change frequency alone is equally poor, since a frequently modified but simple file is not a problem. Their intersection is small, typically a few percent of files, and consistently accounts for a disproportionate share of defects, review time, and development effort. In a legacy system where everything looks bad and the effort available is a fraction of what a full cleanup would require, the value of the analysis is that it answers the question teams cannot otherwise answer: of all this, what do we fix first?

## How to Apply ◆

> The version control history of a long-lived system is an underused record of where the pain actually is, and it is available without instrumenting anything or asking anyone.

- **Extract change frequency per file** from the repository log over a meaningful window — usually one to two years. Shorter windows are noisy; longer ones include churn from a system that no longer exists. Exclude bulk-formatting and rename commits, which otherwise dominate the counts and produce misleading results.
- **Pair frequency with a complexity proxy**: lines of code is crude but works surprisingly well; cyclomatic complexity or indentation depth are better where a tool is available. Plot the two axes and look at the upper-right quadrant. That quadrant is the hotspot set, and it is usually startlingly small.
- **Overlay defect data** by matching commits to bug tickets, if the commit messages or branch names permit it. Files that are frequently changed, complex, and repeatedly implicated in defects are the highest-confidence targets in the codebase.
- Analyze **temporal coupling** — files that repeatedly change in the same commit despite having no explicit dependency. These pairs reveal hidden coupling that no static analysis finds, and they are often the clearest evidence of a missing abstraction or a leaking boundary.
- Look at **author distribution per hotspot**. A hotspot with one contributor is a knowledge risk as well as a code risk; a hotspot with twenty contributors and no owner usually has consistency problems. The two situations call for different responses.
- **Re-run the analysis on a schedule** — quarterly is typical — and track whether the hotspot set is shrinking. A hotspot that has been addressed should fall out of the upper-right quadrant within a couple of cycles; if it does not, the intervention did not work and that is worth knowing.
- Use the output to **direct the improvement budget** rather than to rank teams or individuals. The moment hotspot data is used in performance evaluation, commit behavior changes to optimize the metric and the data stops describing the system.
- Present the analysis **visually to non-technical stakeholders**. A treemap in which size is change frequency and color is complexity communicates the case for maintenance investment far more effectively than any verbal argument about technical debt, because it makes an invisible problem visible.
- **Validate hotspots against the team's experience** before acting. The analysis identifies candidates; developers know which of them are genuinely painful and which are merely large. Where the data and the team disagree, the disagreement is usually informative.

## Tradeoffs ⇄

> Hotspot analysis is cheap, evidence-based, and directs effort well, but it measures proxies rather than quality and can be actively misleading if read naively.

**Benefits:**

- Improvement effort is directed at the small portion of the codebase where it changes anything, rather than distributed evenly across code that is mostly inert.
- Technical debt becomes visible and quantified, which is generally the missing ingredient in conversations with stakeholders about funding maintenance.
- The analysis costs very little — a script over the existing repository history — and requires no cooperation from anyone.
- Temporal coupling reveals architectural problems that no static analysis detects, often identifying exactly where a boundary is missing.
- Progress can be measured over time, so improvement work acquires a metric that is not merely the team's assertion that things are better.

**Costs and Risks:**

- Change frequency measures activity, not quality. A file under active feature development is not necessarily a problem, and treating it as one wastes effort and irritates the team building it.
- Repository history distorts easily: file renames, moves, bulk reformatting, and repository migrations all corrupt the counts unless explicitly handled.
- Lines of code and cyclomatic complexity are weak proxies for the thing that actually matters, which is how hard the code is to change correctly.
- The analysis will not surface a module that is dangerous but rarely touched — including code that is avoided precisely because it is frightening, which is a real blind spot in legacy contexts.
- Used as a performance measure, the data corrupts immediately, since commit granularity and file organization are trivially gamed.

## How It Could Be

A team maintaining a 900,000-line enterprise resource planning system had a technical debt backlog with over 200 entries and no way to order it. They ran a hotspot analysis over 18 months of history and found that 14 files — under one percent of the codebase — accounted for 31 percent of all commits and appeared in 44 percent of commits linked to bug tickets. Four of those files were not on the debt backlog at all, because they were unpleasant rather than obviously broken and nobody had proposed them. The team redirected their improvement budget to the top six hotspots for two quarters. Defect reports attributable to those files fell by roughly two thirds, and the next hotspot run showed all six had dropped out of the upper-right quadrant.

The temporal coupling output from the same analysis produced a more consequential finding. Two files in nominally separate subsystems — an order module and a warehouse module — changed together in 78 percent of the commits that touched either of them, despite no import relationship between them. Investigation found an undocumented shared assumption about a status code enumeration duplicated in both places. It had caused three production incidents over two years, each investigated separately and none of which had identified the pattern. The team unified the enumeration in a week. The visual treemap from this analysis was also what the engineering manager used to secure the following year's maintenance budget, after two previous attempts using verbal arguments had failed.
