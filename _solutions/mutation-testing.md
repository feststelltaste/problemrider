---
title: Mutation Testing
description: Testing the robustness of software tests through targeted code changes
category:
- Testing
quality_tactics_url: https://qualitytactics.de/en/maintainability/mutation-testing
problems:
- poor-test-coverage
- insufficient-testing
- regression-bugs
- legacy-code-without-tests
- quality-blind-spots
- outdated-tests
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Introduce a mutation testing tool appropriate for the project's language (e.g., PIT for Java, Stryker for JavaScript/TypeScript)
- Start with the most critical business logic modules rather than running mutation testing across the entire codebase
- Run mutation testing in CI on changed files or modules to keep feedback loops short
- Use mutation score as a quality indicator alongside code coverage to identify weak test suites
- Focus on surviving mutants: each one represents a test gap that could hide a real bug
- Set incremental mutation score thresholds to gradually improve test quality over time

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reveals tests that pass despite code changes, exposing false confidence in test coverage
- Drives creation of more meaningful, behavior-verifying tests
- Identifies dead code and unreachable branches that mutation testing cannot mutate
- Provides a more accurate quality signal than line coverage alone

**Costs and Risks:**
- Computationally expensive: running hundreds of mutated test cycles takes significant time
- Can produce equivalent mutants that are impossible to detect, creating noise
- May overwhelm teams if applied to large legacy codebases without scoping
- Requires test suites that are already reasonably fast and stable

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A banking application had 85% line coverage, which gave the team confidence in their test suite. When they introduced PIT mutation testing on the loan calculation module, the mutation score was only 42%, meaning more than half of the code mutations went undetected by existing tests. Investigation revealed that many tests were only asserting that methods did not throw exceptions rather than verifying correct output values. The team rewrote the weakest tests and raised the mutation score to 78% within two sprints, catching three previously hidden calculation bugs in the process.
