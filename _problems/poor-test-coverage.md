---
title: Poor Test Coverage
description: Critical parts of the codebase are not covered by tests, creating blind
  spots in quality assurance.
category:
- Code
- Process
- Testing
related_problems:
- slug: quality-blind-spots
  similarity: 0.75
- slug: insufficient-testing
  similarity: 0.6
- slug: testing-complexity
  similarity: 0.6
- slug: inadequate-integration-tests
  similarity: 0.55
- slug: legacy-code-without-tests
  similarity: 0.55
- slug: difficult-to-test-code
  similarity: 0.55
solutions:
- test-coverage-strategy
- code-coverage-analysis
- test-driven-development-tdd
- automated-tests
- acceptance-tests
- functional-tests
- integration-tests
- mutation-testing
- property-based-testing
- behavior-driven-development-bdd
- business-test-cases
- definition-of-done
- platform-independent-test-frameworks
- security-tests
layout: problem
---

## Description

Poor test coverage occurs when significant portions of the codebase, particularly critical functionality, lack adequate automated testing. This creates gaps in quality assurance where bugs can hide undetected until they reach production. Poor coverage doesn't just mean low percentage numbers—it specifically refers to the absence of tests for important business logic, error handling paths, edge cases, and integration points that are crucial for system reliability.

## Indicators ⟡
- Code coverage reports show low percentages, especially in critical business logic areas
- Production bugs frequently occur in areas with little or no test coverage
- Developers are uncertain whether changes will break existing functionality
- Critical system components have no automated tests
- Error handling and edge cases are rarely tested

## Symptoms ▲

- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  Untested code paths allow bugs to reach production undetected, increasing production defect rates.
- [Fear of Change](fear-of-change.md)
<br/>  Without test coverage as a safety net, developers fear making changes that might break untested functionality.
- [Regression Bugs](regression-bugs.md)
<br/>  Lack of automated tests means regressions are not caught during development, appearing later in production.
- [Increased Manual Testing Effort](increased-manual-testing-effort.md)
<br/>  Gaps in automated test coverage must be compensated by extensive manual testing, which is slow and error-prone.
- [Refactoring Avoidance](refactoring-avoidance.md)
<br/>  Without tests to verify correctness, developers avoid refactoring for fear of introducing undetected bugs.
- [Constant Firefighting](constant-firefighting.md)
<br/>  Poor test coverage allows bugs to reach production, which causes constant firefighting.
- [Quality Blind Spots](quality-blind-spots.md)
<br/>  Gaps in test coverage directly create areas where defects go undetected.
## Causes ▼

- [Deadline Pressure](deadline-pressure.md)
<br/>  Time pressure leads teams to skip writing tests in favor of delivering features faster.
- [Difficult to Test Code](difficult-to-test-code.md)
<br/>  Code that is tightly coupled or has hidden dependencies is inherently hard to test, discouraging test creation.
- [Legacy Code Without Tests](legacy-code-without-tests.md)
<br/>  Inherited legacy codebases that lack tests make it very difficult to add coverage incrementally.
- [Inadequate Test Infrastructure](inadequate-test-infrastructure.md)
<br/>  Lack of proper testing tools and infrastructure makes writing and running tests prohibitively difficult.
## Detection Methods ○
- **Code Coverage Analysis:** Use tools to measure what percentage of code is executed by tests
- **Critical Path Identification:** Map business-critical functionality and assess its test coverage
- **Bug Source Analysis:** Track whether production bugs occur in tested vs. untested code areas
- **Coverage Trend Monitoring:** Track whether test coverage is improving, declining, or stagnating over time
- **Manual Testing Dependency:** Identify areas that rely heavily on manual testing due to lack of automation

## Examples

A financial trading application has 40% overall test coverage, but analysis reveals that the core risk calculation algorithms—responsible for preventing catastrophic trading losses—have only 15% test coverage. The existing tests only cover basic scenarios with small trade amounts, but the complex logic handling large trades, margin requirements, and risk limits during market volatility is completely untested. When market conditions change unexpectedly, the untested risk calculation code fails to properly limit exposure, resulting in significant financial losses that could have been prevented by comprehensive testing of edge cases and stress scenarios. Another example involves an e-commerce platform where the payment processing module has 80% line coverage but 0% coverage of error handling paths. While normal payment flows are well-tested, the code that handles declined cards, network timeouts, partial payments, and refund scenarios is never executed by tests. When payment gateway issues occur, customers experience lost transactions, double charges, and failed refunds because the error handling code contains bugs that were never caught during testing.
