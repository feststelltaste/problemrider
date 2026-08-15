---
title: Property-Based Testing
description: Verify software through random inputs and properties
category:
- Testing
problems:
- insufficient-testing
- regression-bugs
- poor-test-coverage
- quality-blind-spots
- legacy-code-without-tests
- increased-risk-of-bugs
- integer-overflow-underflow
- null-pointer-dereferences
- race-conditions
layout: solution
related_solutions:
- slug: mutation-testing
  similarity: 0.75
- slug: automated-tests
  similarity: 0.75
- slug: integration-tests
  similarity: 0.7
- slug: functional-tests
  similarity: 0.7
- slug: test-coverage-strategy
  similarity: 0.7
- slug: cross-version-testing
  similarity: 0.7
---

## Description

Property-based testing generates a large number of random inputs and checks that a general property — an invariant that must hold for all valid inputs, such as idempotency, a round-trip guarantee, or a range constraint — remains true, rather than asserting specific expected outputs for a fixed, hand-picked set of example inputs. When a property fails, the framework's shrinking mechanism automatically reduces the failing input to its smallest reproducing form, which turns a random, possibly large counterexample into a minimal, debuggable test case. This is particularly effective for legacy code because such code is frequently exercised only by a handful of example-based tests written years ago for the scenarios the original author happened to think of, leaving broad swaths of the input space — and the edge cases and boundary conditions the original tests never covered — effectively unverified. Because random input generation actively searches for inputs that violate the stated properties, it routinely surfaces defects that have been present in production code for years without ever being triggered by the narrow set of manually written examples, such as an integer overflow in a rarely used code path. The initial cost is conceptual: articulating properties as universal statements about behavior requires a different mindset than writing example assertions, and not every piece of legacy code has properties that are easy to state, which limits where the technique can be applied without custom input generators for domain-specific types.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A financial services application had a currency conversion module with hand-written tests covering a dozen specific currency pairs. Property-based testing was introduced with properties such as "converting from A to B and back to A should return the original amount within rounding tolerance" and "conversion rates should always be positive." The random generator immediately found a case where converting between two rarely used currencies produced a negative amount due to an integer overflow in an intermediate calculation. This bug had been present for years but was never triggered by the specific test cases the team had written.
