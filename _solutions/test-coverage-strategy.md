---
title: Automated Tests
description: Automatically conduct and regularly execute software tests
category:
- Testing
- Code
quality_tactics_url: https://qualitytactics.de/en/maintainability/automated-tests/
problems:
- legacy-code-without-tests
- poor-test-coverage
- test-debt
- inadequate-integration-tests
- missing-end-to-end-tests
- flaky-tests
- difficult-to-test-code
- regression-bugs
- outdated-tests
- testing-complexity
- testing-environment-fragility
- inadequate-test-data-management
- inadequate-test-infrastructure
- increased-manual-testing-effort
- high-bug-introduction-rate
- high-defect-rate-in-production
- insufficient-testing
layout: solution
---

## How to Apply ◆

> Legacy systems typically operate without automated tests, making every change a gamble against regressions; building test coverage strategically — starting with the highest-risk code rather than pursuing uniform coverage — delivers safety where it matters most without halting delivery.

- Start by writing characterization tests, not correctness tests. In legacy systems the goal is first to capture what the code actually does, including its bugs and undocumented behaviors, so that structural changes can be made safely. Characterization tests record the current behavior; correctness comes later.
- Prioritize test coverage for the code that changes most often and causes the most damage when broken: payment flows, authentication logic, data transformation pipelines, and integration points with external systems. Covering these areas thoroughly provides far more value than achieving uniform coverage across the entire codebase.
- Use the test pyramid deliberately in legacy contexts: invest heavily in unit tests for isolated business logic where seams can be introduced, use integration tests to verify database queries, API calls, and component interactions, and limit end-to-end tests to the most critical user journeys where full-system behavior must be confirmed.
- Apply Michael Feathers' seam techniques to make untested legacy code testable without rewriting it: introduce interfaces at dependency boundaries, use dependency injection to replace hard-coded collaborators with test doubles, and extract logic from framework-coupled classes into plain objects that can be tested in isolation.
- Enforce a no-regression policy: any bug fix must be accompanied by a test that would have caught the bug. This practice builds coverage precisely where the system has historically been fragile and ensures defects are not repeatedly re-introduced.
- Track code coverage as a floor, not a target. Setting a coverage threshold (for example, 60%) that the pipeline enforces prevents coverage from declining as new code is added, while avoiding the trap of writing meaningless tests purely to inflate the percentage.
- Quarantine flaky tests immediately and investigate them as defects. Legacy systems often have nondeterministic behavior caused by global state, timing dependencies, or inconsistent test data; a flaky test that the team learns to ignore destroys trust in the entire suite.
- Create and maintain a dedicated test data strategy: avoid shared mutable test data that causes tests to interfere with each other, and build factory functions or builder patterns that create isolated, self-describing test records. Legacy databases with production-derived test data are a common source of test fragility.

## Tradeoffs ⇄

> Automated test coverage transforms a legacy system from a fragile artifact that cannot be safely changed into a codebase that developers can refactor, extend, and deploy with confidence.

**Benefits:**

- Provides the safety net that makes refactoring possible in a legacy codebase: without tests, improving code structure is a gamble; with tests, each transformation can be verified as behavior-preserving.
- Catches regressions immediately after the change that caused them, when the developer still has full context, rather than weeks later in a manual test cycle.
- Reduces the manual testing burden on each release, freeing QA time for exploratory testing of new behavior rather than re-verification of existing functionality that automation can handle.
- Serves as executable documentation of the legacy system's actual behavior, complementing or replacing written specifications that have drifted out of sync with the code over years.
- Enables the CI/CD pipeline to provide meaningful deployment confidence, turning automated tests into the primary quality gate rather than a supplementary check.

**Costs and Risks:**

- Legacy code that was not designed for testability — with global state, hard-coded dependencies, and framework-coupled logic — requires significant restructuring before unit tests can be written, making the initial investment higher than for greenfield systems.
- Characterization tests that capture current bugs and incorrect behaviors must be carefully managed: they are useful during structural refactoring but must be updated or replaced when those bugs are eventually fixed, or they will lock in incorrect behavior permanently.
- Achieving meaningful test coverage on a large legacy codebase takes months or years of consistent effort; teams that expect quick results may abandon the practice before the safety net is strong enough to change behavior.
- Poorly written tests — testing implementation details rather than behavior, or making assertions so broad they never fail — create false confidence while adding maintenance overhead without providing real protection.
- Slow test suites that take hours to run are common in legacy systems where integration tests accumulate against real databases and external systems; without active management, the pipeline becomes a bottleneck that teams work around rather than through.

## Examples

> The following scenarios illustrate how a deliberate test coverage strategy brings a legacy system under control without requiring a disruptive stop-the-world testing effort.

A logistics software company inherited a legacy route optimization system from an acquisition. The system had zero automated tests and a history of regressions with each new release, causing the new owners to halt all feature development while manually verifying every change. Rather than attempting comprehensive coverage, the team spent their first month identifying the five most frequently broken behaviors in the previous twelve months of bug reports: toll calculation, weight limit enforcement, hazardous material routing rules, multi-stop sequencing, and time window validation. They wrote characterization tests for each of these five areas, covering the inputs and outputs observed in production logs. Within six weeks they had 340 tests covering the highest-risk behaviors. Regressions in these areas dropped to zero within the quarter, and the team resumed feature development with a targeted safety net in place.

A European bank operating a legacy trade settlement system had a test suite of 4,000 end-to-end tests that took eight hours to run. The suite was nominally comprehensive but so slow and brittle that the team ran it only once per week and ignored individual failures unless they recurred. The safety net existed on paper but provided no practical protection. A newly hired test architect analyzed the suite and found that 90% of the scenarios were covered multiple times by different end-to-end tests. The team spent three months replacing redundant end-to-end tests with focused integration tests and targeted unit tests for the business logic those end-to-end tests had been verifying. The new suite of 1,200 tests ran in under twenty minutes. With the pipeline now fast enough to run on every pull request, the team discovered and fixed three regressions in the first month that the previous weekly run would have caught too late to be attributed to their causing change.

An insurance company maintaining a legacy claims processing system faced a critical modernization: replacing the underlying rules engine with a newer platform. The claims logic had never been tested automatically, and the team was paralyzed by fear that the replacement would silently alter the calculation of claim amounts. Before touching any production code, the team spent two months building a characterization test suite by running thousands of historical claims through the existing system and recording the outputs. These outputs became the expected values for the same claims run through the new rules engine. The characterization tests identified 23 calculation differences between the old and new engines during parallel testing, 19 of which were genuine bugs in the legacy engine that were corrected in the new implementation and 4 of which were intentional behavioral differences requiring business sign-off. Without the characterization test suite, the migration would have been impossible to validate; with it, the cutover was executed with documented confidence.
