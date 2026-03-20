---
title: Plausibility Checks
description: Checking inputs, data, or states for validity to detect potential errors early
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/reliability/plausibility-checks
problems:
- silent-data-corruption
- inadequate-error-handling
- unpredictable-system-behavior
- data-migration-integrity-issues
- increased-error-rates
- brittle-codebase
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Add input validation at system boundaries to reject obviously invalid data before it enters the processing pipeline
- Implement range checks, format validations, and business rule assertions on critical data fields
- Add cross-field consistency checks that verify related data values are plausible together
- Introduce output validation that verifies results fall within expected ranges before returning them to callers
- Place plausibility checks at data import and migration points to catch errors during legacy data transitions
- Log and alert on plausibility violations to provide early warning of data quality issues
- Use defensive assertions in critical calculation paths to catch unexpected intermediate states

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Catches data errors early before they propagate through the system and become harder to diagnose
- Prevents silent corruption that can cause incorrect business outcomes
- Provides clear error messages that help developers and users identify the source of problems
- Improves confidence in data quality during migrations and system integrations

**Costs and Risks:**
- Overly strict checks can reject valid edge cases, especially in legacy data with historical anomalies
- Adding checks to hot paths may introduce measurable performance overhead
- Maintaining plausibility rules requires ongoing updates as business rules evolve
- Legacy data may contain historical values that violate newly introduced rules

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A utility company's legacy billing system occasionally generated invoices with negative amounts or impossibly high consumption values due to meter reading errors and data conversion bugs. By adding plausibility checks that validated consumption values against historical ranges and flagged invoices exceeding configurable thresholds, the team caught 95% of billing errors before they reached customers. The checks also uncovered a long-standing unit conversion bug in the data import module that had been causing subtle overcharges for a subset of commercial customers.
