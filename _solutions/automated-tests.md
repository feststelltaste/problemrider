---
title: Automated Tests
description: Automated verification of functionality at various levels
category:
- Testing
- Code
quality_tactics_url: https://qualitytactics.de/en/reliability/automated-tests
problems:
- insufficient-testing
- poor-test-coverage
- legacy-code-without-tests
- regression-bugs
- fear-of-change
- high-bug-introduction-rate
- increased-manual-testing-effort
- difficult-to-test-code
- test-debt
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Start by adding characterization tests that document the current behavior of the legacy system before making any changes
- Identify high-risk areas of the codebase using defect history and change frequency, and prioritize test coverage there
- Introduce unit tests for new code and code being modified, following the Boy Scout Rule of leaving code better than you found it
- Add integration tests at module boundaries to verify interactions between components
- Use approval testing or snapshot testing for complex legacy outputs where writing assertion-based tests is impractical
- Set up a CI pipeline that runs tests automatically on every commit to provide rapid feedback
- Establish minimum coverage thresholds for new code while gradually increasing overall coverage targets
- Refactor tightly coupled code to improve testability incrementally, extracting dependencies behind interfaces

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides a safety net that enables confident refactoring and modernization of legacy code
- Catches regression bugs early before they reach production
- Reduces the need for expensive manual testing cycles
- Serves as living documentation of expected system behavior
- Accelerates development velocity over time by reducing debugging effort

**Costs and Risks:**
- Writing tests for legacy code without clear interfaces requires significant initial investment
- Poorly written tests can become a maintenance burden themselves
- High test coverage does not guarantee absence of bugs if tests are superficial
- Teams unfamiliar with testing practices need training and mentoring
- Slow test suites can become a bottleneck if not properly structured and maintained

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A banking application had zero automated tests and relied entirely on a four-week manual regression testing cycle before each release. The team began by writing characterization tests for the payment processing module using recorded production transactions. Over six months, they built up 800 tests covering the critical path. Releases that previously required weeks of manual testing could be validated in 20 minutes of automated test execution. Regression defects in the payment module dropped by 75%, and the team gained the confidence to begin refactoring the most problematic areas of the codebase.
