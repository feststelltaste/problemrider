---
title: Value Range Definition
description: Define acceptable value ranges for inputs and outputs
category:
- Code
- Testing
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/value-range-definition
problems:
- inadequate-error-handling
- inconsistent-behavior
- unpredictable-system-behavior
- silent-data-corruption
- hardcoded-values
- regression-bugs
- increased-risk-of-bugs
layout: solution
---

## How to Apply ◆

> In legacy systems, value ranges are often enforced inconsistently or not at all — making them explicit prevents data corruption and catches integration errors at system boundaries.

- Audit the legacy system to identify all input and output values, documenting the actual ranges observed in production data rather than relying on potentially outdated documentation.
- Define explicit validation rules for every system boundary — API endpoints, user interfaces, file imports, database writes — rejecting values outside acceptable ranges with clear error messages.
- Pay special attention to values that the legacy system accepted silently but handled incorrectly, such as negative amounts in fields that should only be positive or dates outside business-meaningful ranges.
- Implement validation as close to the system boundary as possible to prevent invalid data from propagating through the system.
- Create comprehensive test cases for boundary values, including minimum, maximum, just-inside, just-outside, and null/empty values for each defined range.
- Document value range decisions and their rationale, especially when the replacement system's ranges differ from the legacy system's implicit behavior.

## Tradeoffs ⇄

> Explicit value ranges prevent data corruption and clarify system behavior but require effort to define correctly and may reject data that the legacy system accepted.

**Benefits:**

- Prevents silent data corruption by catching invalid values at system boundaries before they enter the processing pipeline.
- Makes system behavior predictable and documented rather than relying on implicit assumptions about valid data.
- Simplifies debugging by ensuring that internal processing only handles values within known-good ranges.
- Provides clear documentation of system constraints for API consumers and integration partners.

**Costs and Risks:**

- Defining ranges for a legacy system with years of accumulated data may reveal that production data already contains out-of-range values that must be cleaned up or grandfathered.
- Overly restrictive ranges can reject legitimate data that the legacy system accepted, causing user frustration and workflow disruption.
- Maintaining value range definitions as business rules change requires ongoing governance.
- Validation logic can become scattered across the codebase if not centralized in a validation layer.

## How It Could Be

> The following scenario shows how value range definition prevents data quality issues during legacy migration.

A supply chain company discovered during migration that its legacy inventory system accepted negative stock quantities, which had been used as an informal back-ordering mechanism for 10 years. The new system's strict non-negative validation initially rejected these entries, breaking the ordering workflow. By defining an explicit value range policy — stock quantities must be non-negative, with a separate back-order quantity field — the team preserved the business capability while eliminating the data ambiguity that had caused reporting errors for years. The migration required a data cleanup step that converted 8,000 negative stock records into proper back-order entries, finally making inventory reports accurate.
