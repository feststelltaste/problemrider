---
title: Test-Driven Development (TDD)
description: Writing tests before the actual implementation
category:
- Code
- Testing
quality_tactics_url: https://qualitytactics.de/en/maintainability/test-driven-development-tdd
problems:
- legacy-code-without-tests
- poor-test-coverage
- regression-bugs
- difficult-to-test-code
- fear-of-change
- high-bug-introduction-rate
- refactoring-avoidance
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Follow the Red-Green-Refactor cycle: write a failing test, make it pass with minimal code, then refactor
- When modifying legacy code, write characterization tests first to capture existing behavior before making changes
- Start applying TDD to new code and bug fixes rather than attempting to retrofit the entire legacy codebase
- Use TDD as a design tool: if code is hard to test, the design likely needs improvement
- Keep test cycles short (under a few minutes per Red-Green-Refactor cycle) to maintain flow
- Pair TDD with refactoring: after tests pass, improve the design while the safety net is in place
- Build team skills through coding dojos and pair programming sessions focused on TDD

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Produces code with built-in test coverage from the start
- Drives simpler, more modular designs because testability is a design constraint
- Provides immediate feedback on whether code changes break existing behavior
- Reduces debugging time by catching defects at the moment they are introduced

**Costs and Risks:**
- Requires significant practice to become proficient; initial productivity may decrease
- Not all legacy code is amenable to TDD without first extracting dependencies
- Can lead to over-testing of implementation details if properties are not well chosen
- Teams under heavy deadline pressure may abandon the practice before realizing benefits

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A team maintaining a legacy payroll system was tasked with adding support for a new tax regulation. Rather than modifying the existing untested tax calculation code directly, they wrote characterization tests to capture the current behavior of the module. Then, using TDD, they wrote failing tests for the new regulation, implemented the logic to make them pass, and refactored the result. The process took slightly longer than the team's usual approach, but the module shipped with comprehensive test coverage. When a follow-up regulation change arrived two months later, the team made the modification confidently in half the time, guided by the existing test suite.
