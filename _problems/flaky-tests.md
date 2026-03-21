---
title: Flaky Tests
description: Tests fail randomly due to timing, setup, or dependencies, undermining
  trust in the test suite
category:
- Code
- Process
related_problems:
- slug: testing-environment-fragility
  similarity: 0.7
- slug: outdated-tests
  similarity: 0.6
- slug: testing-complexity
  similarity: 0.55
- slug: difficult-to-test-code
  similarity: 0.55
- slug: quality-blind-spots
  similarity: 0.55
- slug: inadequate-integration-tests
  similarity: 0.55
solutions:
- isolated-test-environments
- test-coverage-strategy
layout: problem
---

## Description

Flaky tests are automated tests that produce inconsistent results when run multiple times against the same code, sometimes passing and sometimes failing without any changes to the codebase. These tests undermine confidence in the entire test suite, making it difficult to distinguish between real regressions and false positives. Over time, teams begin to ignore test failures or disable flaky tests, reducing the effectiveness of automated testing as a safety net for code changes.

## Indicators ⟡

- Tests that occasionally fail on continuous integration but pass when run locally
- Team members regularly re-running failed test suites to see if they pass the second time
- Tests that fail more frequently during high system load or specific times of day
- Intermittent test failures that are difficult to reproduce consistently
- Tests that depend on external services or network connectivity
- Test setup or teardown processes that don't consistently reset system state
- Tests with hard-coded timing assumptions or sleep statements

## Symptoms ▲

- [Increased Manual Testing Effort](increased-manual-testing-effort.md)
<br/>  When automated tests are unreliable, teams compensate by increasing manual testing to catch regressions.
- [Slow Development Velocity](slow-development-velocity.md)
<br/>  Developers waste time re-running test suites, investigating false failures, and losing confidence in automated testing.
- [Quality Blind Spots](quality-blind-spots.md)
<br/>  When flaky tests are disabled or ignored, they create gaps in test coverage where real bugs can hide undetected.
- [Review Bottlenecks](review-bottlenecks.md)
<br/>  CI pipelines blocked by flaky test failures delay code review and merge processes.
- [Long Build and Test Times](long-build-and-test-times.md)
<br/>  Flaky tests lead to longer build times because developers re-run test suites multiple times and CI pipelines get bloc....
## Causes ▼

- [Testing Environment Fragility](testing-environment-fragility.md)
<br/>  Unreliable testing infrastructure causes tests to produce different results depending on environment conditions.
- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  Tests coupled to external services, shared state, or other tests produce inconsistent results due to environmental dependencies.
- [Difficult to Test Code](difficult-to-test-code.md)
<br/>  Code that is hard to test in isolation forces tests to depend on timing, external services, or shared state, creating flakiness.
- [Inadequate Test Data Management](inadequate-test-data-management.md)
<br/>  Unrealistic or inconsistent test data causes tests to produce different results across runs.
## Detection Methods ○

- Track test failure rates and patterns over time to identify inconsistent tests
- Run test suites multiple times in succession to identify non-deterministic behavior
- Monitor CI/CD pipeline metrics for tests that fail and then pass without code changes
- Use test flakiness detection tools that analyze historical test results
- Implement test quarantine systems that flag unreliable tests
- Review test code for timing dependencies, external service calls, and shared state
- Analyze test failures by time of day, system load, or environmental factors

## Examples

A web application's test suite includes an integration test that verifies user registration functionality. The test creates a user account, sends a verification email, and checks that the account becomes active. However, the test sometimes fails because it doesn't wait long enough for the email service to process the request before checking the account status. On fast test environments, the test passes, but on slower systems or during high load, the email processing takes longer and the test fails. The team initially ignores these failures as "environment issues," but over time, more tests develop similar timing problems. Eventually, the team loses confidence in the test suite and begins relying more heavily on manual testing, missing several real bugs that automated tests could have caught.
