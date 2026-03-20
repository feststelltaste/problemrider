---
title: Regression Testing
description: Re-running existing tests after every change against unintended breakage
category:
- Testing
- Code
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/regression-testing/
problems:
- increased-bug-count
- increased-risk-of-bugs
- delayed-bug-fixes
- maintenance-paralysis
- large-estimates-for-small-changes
- reduced-code-submission-frequency
- rapid-system-changes
- increased-cost-of-development
- slow-development-velocity
- customer-dissatisfaction
- user-trust-erosion
layout: solution
---

## How to Apply ◆

> Legacy systems without regression testing operate in a state of permanent uncertainty: every change might break something, but the team has no way to know until users report problems. Introducing regression testing into a legacy codebase is not about achieving 100% coverage from day one — it is about systematically building a safety net that enables confident modification of the most critical and frequently changed areas.

- Begin by identifying the system's most critical user-facing workflows and writing end-to-end regression tests that verify these workflows produce correct results. In legacy systems, even a small suite of tests covering the top ten user workflows provides more value than extensive unit tests in rarely modified code, because it directly addresses the user trust erosion caused by changes breaking visible functionality.
- Write characterization tests for the areas of code that the team needs to modify but is afraid to touch. Characterization tests capture the current behavior of the system — including bugs and undocumented side effects — as a baseline. They do not assert that the behavior is correct; they assert that it has not changed, which is exactly the safety net needed to overcome maintenance paralysis.
- Integrate regression tests into the CI pipeline so they run automatically on every pull request. This integration is the mechanism that converts tests from a manual verification step (which gets skipped under pressure) into a mandatory quality gate. For legacy systems where the full test suite is slow, run a fast critical-path subset on every PR and the full suite nightly.
- When fixing a bug, write a regression test that reproduces the bug before implementing the fix. This practice ensures that fixed bugs stay fixed, directly addressing the delayed bug fixes problem where the same issues reappear after being resolved. Over time, this approach builds a comprehensive regression suite focused on the areas of the system that actually produce defects.
- Establish a policy that no code is merged without passing all existing regression tests, and make test failures a blocking condition in the review process. This policy directly encourages more frequent code submissions because developers receive fast, automated feedback on whether their changes break anything, removing the fear that drives reduced code submission frequency.
- For legacy systems undergoing rapid changes, implement regression testing at the API contract level in addition to unit and integration levels. API-level regression tests catch breaking changes that affect consumers without depending on the internal implementation, which may be evolving rapidly.
- Track regression test coverage growth over time as a metric that demonstrates progress toward safer modifiability. Report coverage in terms of critical workflows covered rather than lines of code, because stakeholders understand "we can now safely modify the payment workflow" better than "we reached 45% line coverage."
- Allocate dedicated time in each sprint for writing regression tests around code that is about to be modified, treating test creation as a prerequisite for the modification rather than an afterthought. This approach ensures that test coverage grows in the areas where it is most needed, driven by actual development activity rather than abstract coverage targets.

## Tradeoffs ⇄

> Regression testing converts the invisible risk of legacy system modification into visible, manageable verification, enabling the team to move from fearful avoidance to confident improvement, but requires sustained investment in test creation and maintenance.

**Benefits:**

- Directly addresses maintenance paralysis by providing the confidence to modify code that the team has been afraid to touch, breaking the cycle where fear of breaking things prevents necessary improvements.
- Reduces large estimates for small changes by enabling developers to verify the impact of modifications quickly rather than requiring extensive manual testing and risk analysis before every change.
- Decreases increased bug count by catching regressions before they reach production, preventing the compounding effect where each release introduces new defects that erode quality and user trust.
- Enables more frequent code submissions by providing fast automated feedback that replaces the slow, manual verification that discourages developers from submitting incremental changes.
- Reduces increased cost of development by catching bugs early in the development cycle when they are cheaper to fix, rather than discovering them in production where diagnosis and repair are expensive.
- Rebuilds user trust erosion by reducing the frequency of visible regressions that damage users' confidence in the system's reliability.

**Costs and Risks:**

- Writing regression tests for legacy systems with tightly coupled components and no dependency injection is genuinely difficult and may require refactoring the code to make it testable, creating a chicken-and-egg problem.
- Regression test suites that are not maintained become brittle and produce false failures, training developers to ignore test results rather than trust them, which is worse than having no tests at all.
- Slow regression test suites that take hours to complete can become bottlenecks that delay releases and frustrate developers, requiring investment in test infrastructure, parallelization, and selective test execution.
- Characterization tests that capture buggy behavior create a tension when the team later wants to fix those bugs: the tests must be updated alongside the fixes, which requires understanding which test assertions represent intentional behavior and which represent bugs.
- Over-reliance on end-to-end regression tests without complementary unit and integration tests creates a test suite that is slow to run, expensive to maintain, and provides poor diagnostic information when failures occur.

## How It Could Be

> The following scenarios illustrate how regression testing enables confident modification of legacy systems where fear of breaking things has previously paralyzed improvement efforts.

A government agency operates a legacy tax calculation system that processes millions of returns annually. The system has no automated tests, and the development team has been unable to update the tax rules engine for two years because every previous modification attempt caused incorrect calculations that were only discovered after returns had been processed. The team begins by writing characterization tests that capture the output of the existing rules engine for a representative set of 5,000 historical tax returns, covering all major filing categories and edge cases. These tests do not assert that the calculations are correct — they assert that the results match the current system's output. With this safety net in place, the team modifies the rules engine to support new tax legislation, running the characterization tests after each change to verify that only the intended calculations changed. The update that had been blocked for two years is completed in six weeks, with zero post-deployment calculation errors. The characterization test suite becomes a permanent asset that enables annual tax rule updates with confidence.

A SaaS company's development team has seen their code submission frequency drop from daily pull requests to weekly batches because developers are afraid that their changes will break the application and trigger an emergency production fix. Analysis reveals that the previous five production incidents were all regressions — features that previously worked correctly but were broken by seemingly unrelated changes. The team implements a regression testing strategy focused on the specific integration points where regressions have historically occurred. For each past regression, they write a test that would have caught it, and they add regression tests for any new bug fix before merging the fix. Within three months, the regression test suite covers 85% of the integration points that historically produce failures. Developer confidence increases measurably: code submissions return to a daily cadence, pull request sizes shrink from an average of 400 lines to 80 lines, and code review quality improves because reviewers can focus on design rather than worrying about hidden breakage. The number of production regressions drops from five per quarter to zero over two consecutive quarters.
