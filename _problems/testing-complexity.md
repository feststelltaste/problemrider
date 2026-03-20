---
title: Testing Complexity
description: Quality assurance must verify the same functionality in multiple locations,
  which increases the testing effort and the risk of missing bugs.
category:
- Code
- Testing
related_problems:
- slug: difficult-to-test-code
  similarity: 0.7
- slug: insufficient-testing
  similarity: 0.65
- slug: complex-and-obscure-logic
  similarity: 0.6
- slug: test-debt
  similarity: 0.6
- slug: inconsistent-quality
  similarity: 0.6
- slug: quality-blind-spots
  similarity: 0.6
solutions:
- test-coverage-strategy
layout: problem
---

## Description
Testing complexity is a common problem in software systems with a high degree of code duplication. It occurs when quality assurance (QA) must verify the same functionality in multiple locations. This increases the testing effort and the risk of missing bugs. Testing complexity is often a sign of a poorly designed system with a high degree of code duplication.

## Indicators ⟡
- The QA team is spending a lot of time testing the same functionality over and over again.
- The QA team is not able to keep up with the pace of development.
- The QA team is missing a lot of bugs.
- The QA team is not happy with the quality of the system.

## Symptoms ▲

- [Insufficient Testing](insufficient-testing.md)
<br/>  The high effort required to test duplicated functionality leads to insufficient test coverage overall.
- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  The increased risk of missing bugs due to duplicated test surfaces means more issues reach production.
- [Delayed Value Delivery](delayed-value-delivery.md)
<br/>  The QA team's inability to keep up with development due to testing overhead slows overall delivery.
- [Inconsistent Quality](inconsistent-quality.md)
<br/>  Some instances of duplicated functionality get thoroughly tested while others are missed, resulting in uneven quality.
## Causes ▼

- [Code Duplication](code-duplication.md)
<br/>  Duplicated code means the same functionality must be verified in multiple locations, directly multiplying testing effort.
- [Complex and Obscure Logic](complex-and-obscure-logic.md)
<br/>  Complex, hard-to-understand logic requires more elaborate test scenarios and makes it harder to achieve adequate coverage.
- [Difficult to Test Code](difficult-to-test-code.md)
<br/>  Code that is inherently difficult to test due to poor design adds to overall testing complexity.
- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  Tightly coupled components cannot be tested in isolation, requiring complex integration test setups.
- [Test Debt](test-debt.md)
<br/>  When testing is too complex, teams take shortcuts and skip tests, accumulating test debt over time.
## Detection Methods ○
- **Test Case Analysis:** Analyze your test cases to identify duplicated tests.
- **Code Coverage Analysis:** Analyze your code coverage to identify areas of the system that are not being tested.
- **QA Team Feedback:** Listen to feedback from the QA team to identify areas of the system that are difficult to test.
- **Bug Triage:** Analyze your bug triage process to identify bugs that are being missed by the QA team.

## Examples
An e-commerce website has a checkout flow that is duplicated in two different places. The QA team has to test the checkout flow in both places to make sure that it is working correctly. This is a waste of time and effort, and it increases the risk of missing bugs. The problem could be solved by creating a single, reusable checkout flow that is used in both places.
