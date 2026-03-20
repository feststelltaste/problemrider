---
title: Property-Based Testing
description: Verify software through random inputs and properties
category:
- Testing
quality_tactics_url: https://qualitytactics.de/en/maintainability/property-based-testing
problems:
- insufficient-testing
- regression-bugs
- poor-test-coverage
- quality-blind-spots
- legacy-code-without-tests
- increased-risk-of-bugs
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify pure functions and data transformations in the legacy code that have well-defined properties (e.g., idempotency, reversibility, invariants)
- Use a property-based testing framework appropriate for the language (e.g., QuickCheck, jqwik, Hypothesis, fast-check)
- Define properties as universal truths about the code rather than specific input-output pairs
- Start with serialization/deserialization round-trip tests and mathematical properties as easy wins
- Use shrinking capabilities to automatically find the minimal failing input when a property violation is discovered
- Combine property-based tests with traditional example-based tests for comprehensive coverage

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Discovers edge cases and boundary conditions that developers would not think to test manually
- Provides broader coverage than hand-written example tests with fewer test cases to maintain
- Shrinking automatically produces minimal reproduction cases, simplifying debugging
- Forces developers to think about invariants and contracts rather than specific scenarios

**Costs and Risks:**
- Writing good properties requires a different mindset and can be initially challenging for teams
- Random generation may not produce relevant inputs without custom generators for domain types
- Flaky results can occur if properties are not deterministic or if seed management is neglected
- Not all legacy code has easily expressible properties, limiting applicability

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A financial services application had a currency conversion module with hand-written tests covering a dozen specific currency pairs. Property-based testing was introduced with properties such as "converting from A to B and back to A should return the original amount within rounding tolerance" and "conversion rates should always be positive." The random generator immediately found a case where converting between two rarely used currencies produced a negative amount due to an integer overflow in an intermediate calculation. This bug had been present for years but was never triggered by the specific test cases the team had written.
