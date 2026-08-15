---
title: Acceptance Tests
description: Verify fulfillment of business requirements through automated tests
category:
- Testing
- Requirements
problems:
- insufficient-testing
- poor-test-coverage
- missing-end-to-end-tests
- regression-bugs
- legacy-code-without-tests
- fear-of-change
- inadequate-requirements-gathering
- increased-manual-testing-effort
- reduced-feature-quality
layout: solution
related_solutions:
- slug: automated-tests
  similarity: 0.8
- slug: functional-tests
  similarity: 0.75
- slug: test-coverage-strategy
  similarity: 0.75
- slug: behavior-driven-development-bdd
  similarity: 0.75
- slug: user-acceptance-tests
  similarity: 0.75
- slug: specification-by-example
  similarity: 0.75
---

## Description

Acceptance tests are automated tests that verify a system fulfills its business requirements from the perspective of an end user or business stakeholder, rather than checking implementation details the way unit tests do. They are typically written against a business-readable specification format, using tools such as Cucumber, FitNesse, or Robot Framework, so that the same scenario definitions can be understood, reviewed, and even authored by non-developers. In legacy systems that were never covered by automated tests, acceptance tests fill a specific and urgent gap: they capture what the system is currently expected to do at the level that matters most to the business, giving the team a safety net before touching code whose internal behavior nobody fully understands anymore. This makes acceptance tests a prerequisite for safe modernization work such as extracting modules, replacing components, or migrating platforms, because a passing acceptance suite is direct evidence that a change has not altered externally visible business behavior. Building this suite for an existing legacy system requires significant upfront investment, since the tests must be written retroactively for functionality that already exists rather than test-first alongside new development, and it also requires close collaboration with domain experts who can confirm the tests reflect actual business intent rather than assumed behavior. Over time the suite doubles as executable documentation of the system's behavior, often the most reliable documentation the legacy system has.

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
