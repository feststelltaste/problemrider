---
title: Acceptance Tests
description: Verify fulfillment of business requirements through automated tests
category:
- Testing
- Requirements
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/acceptance-tests
problems:
- insufficient-testing
- poor-test-coverage
- missing-end-to-end-tests
- regression-bugs
- legacy-code-without-tests
- fear-of-change
- inadequate-requirements-gathering
layout: solution
---

## How to Apply ◆

- Define acceptance criteria for each business requirement and translate them into automated test cases before or alongside implementation.
- Use frameworks like Cucumber, FitNesse, or Robot Framework that allow business stakeholders to read and validate test scenarios.
- Start with the most critical legacy workflows: identify the top business processes and create acceptance tests that verify their correct behavior.
- Run acceptance tests as part of the CI/CD pipeline to catch regressions before deployment.
- Use acceptance tests as a safety net before refactoring legacy code, ensuring existing behavior is preserved.
- Involve domain experts in reviewing and authoring test scenarios to ensure tests reflect actual business intent.

## Tradeoffs ⇄

**Benefits:**
- Provides confidence that business requirements are met after changes to legacy code.
- Creates executable documentation of expected system behavior.
- Bridges the gap between business stakeholders and developers by using shared test language.
- Enables safer refactoring and modernization by detecting functional regressions.

**Costs:**
- Writing acceptance tests for existing legacy functionality requires significant upfront investment.
- Tests can become brittle if they depend on UI elements or specific implementation details.
- Maintaining a large suite of acceptance tests requires ongoing effort as requirements evolve.
- Slow execution times for comprehensive acceptance test suites can delay feedback.

## How It Could Be

A retail company inherits a legacy order management system with no automated tests. Before beginning modernization, the team collaborates with business analysts to identify the twenty most critical order workflows and writes acceptance tests for each using Cucumber. These tests verify end-to-end behavior including order creation, payment processing, inventory updates, and notification delivery. When the team later extracts the payment module into a separate service, the acceptance tests catch three subtle regressions in discount calculation logic that unit tests would not have detected. The test suite becomes the definitive specification of correct behavior, referenced by both developers and business stakeholders during planning discussions.
