---
title: Code Coverage Analysis
description: Measurement of the proportion of code covered by tests
category:
- Testing
quality_tactics_url: https://qualitytactics.de/en/maintainability/code-coverage-analysis
problems:
- poor-test-coverage
- legacy-code-without-tests
- insufficient-testing
- fear-of-breaking-changes
- regression-bugs
- test-debt
- quality-blind-spots
layout: solution
---

## How to Apply ◆

> In legacy systems, code coverage analysis reveals which parts of the codebase are protected by tests and which are blind spots where changes carry the highest risk.

- Integrate a coverage analysis tool (JaCoCo, Istanbul, coverage.py) into the CI pipeline to measure coverage on every build and track trends over time.
- Use coverage data to identify the riskiest parts of the legacy codebase — modules with high change frequency but low test coverage are the highest priority for test investment.
- Set realistic coverage targets that increase incrementally rather than demanding immediate high coverage on a legacy codebase that may start near zero.
- Enforce a coverage ratchet that prevents coverage from decreasing — new changes must maintain or improve the overall coverage percentage.
- Distinguish between line coverage, branch coverage, and mutation testing results — line coverage alone can create false confidence when conditional logic is not fully tested.
- Use coverage reports during code reviews to verify that new code and modified legacy code include appropriate test coverage.
- Focus coverage improvement efforts on business-critical paths and frequently modified code rather than pursuing a uniform coverage percentage across the entire codebase.

## Tradeoffs ⇄

> Coverage analysis identifies testing gaps but can become a misleading metric if pursued as a goal in itself rather than a tool for risk management.

**Benefits:**

- Makes testing gaps visible, enabling the team to prioritize test investment where it will have the greatest risk-reduction impact.
- Provides an objective metric for tracking testing improvement over time during legacy modernization.
- Helps identify dead code — code with zero coverage that is also never reached in production may be safe to remove.
- Prevents coverage regression by alerting the team when changes reduce the proportion of tested code.

**Costs and Risks:**

- High coverage numbers can create false confidence — 80% line coverage does not mean 80% of behaviors are tested if edge cases and error paths are uncovered.
- Pursuing coverage as a target rather than a tool can lead to low-value tests that increase the coverage number without meaningfully reducing risk.
- Coverage analysis adds time to the build process, which may be unwelcome in legacy systems with already-long build times.
- Focus on coverage metrics can divert attention from test quality — a few well-designed tests often provide more protection than many superficial ones.

## How It Could Be

> The following scenario demonstrates how coverage analysis guides testing investment in a legacy codebase.

A fintech company's legacy payment processing system had 12% overall test coverage. Rather than setting a blanket 80% coverage target, the team used coverage analysis combined with change frequency data to identify the 30 classes that were both frequently modified and had zero test coverage. These classes accounted for 60% of production defects. By focusing test investment on these high-risk classes first, the team increased coverage to 25% overall but covered 85% of the frequently changed code. Over the following year, production defects in the targeted classes dropped by 70%, validating the strategy of using coverage data for risk-based test prioritization rather than uniform coverage goals.
