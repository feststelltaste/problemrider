---
title: Integration Tests
description: Conduct tests to verify the interaction of different system components
category:
- Testing
quality_tactics_url: https://qualitytactics.de/en/maintainability/integration-tests
problems:
- inadequate-integration-tests
- missing-end-to-end-tests
- regression-bugs
- fear-of-change
- deployment-risk
- cascade-failures
- integration-difficulties
- poor-test-coverage
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify the most critical integration points: database access, external service calls, message queues, and inter-module boundaries
- Start with the integration seams that have caused the most production incidents
- Use test containers or embedded databases to create repeatable, isolated integration test environments
- Write tests that exercise the real interaction path rather than mocking away the integration
- Keep integration tests focused on verifying contracts between components, not on testing business logic
- Automate integration tests as part of the CI pipeline so they run on every commit
- Maintain a separate integration test suite with clear naming conventions to distinguish from unit tests

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Catches bugs at component boundaries that unit tests cannot detect
- Increases confidence when modifying legacy code that touches multiple subsystems
- Provides a safety net for refactoring and migration efforts
- Documents how components are expected to interact

**Costs and Risks:**
- Integration tests are slower than unit tests and can slow down the feedback loop if not managed
- Test environment setup and maintenance adds ongoing effort
- Flaky tests due to timing, network, or state issues can erode trust in the test suite
- May create a false sense of security if only happy paths are tested

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A retail company had a legacy order processing system where the order service, inventory service, and payment gateway were tightly integrated but had no integration tests. Every release was a gamble because unit tests passed but production failures at integration boundaries were common. The team introduced integration tests using Testcontainers for the database and WireMock for the payment gateway. These tests caught a critical serialization mismatch between the order and inventory services that had been causing silent data loss. After establishing the integration test suite, the team reduced production incidents related to component interactions by over 60%.
