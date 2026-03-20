---
title: Decision Tables
description: Define and evaluate complex business rules in tabular form
category:
- Code
- Requirements
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/decision-tables
problems:
- complex-and-obscure-logic
- legacy-business-logic-extraction-difficulty
- difficult-code-comprehension
- poor-domain-model
- requirements-ambiguity
- spaghetti-code
layout: solution
---

## How to Apply ◆

- Identify complex conditional logic in the legacy codebase (deeply nested if-else chains, switch statements with many cases) that represents business rules.
- Extract these business rules into decision tables that map input conditions to expected outputs or actions.
- Have business stakeholders validate the decision tables to confirm they accurately represent the intended business logic.
- Implement decision tables using a rules engine (Drools, Easy Rules) or simple table-driven code that replaces the complex conditional logic.
- Write test cases derived from the decision table rows to verify that the implementation matches the specification.
- Maintain decision tables as living documentation that is updated when business rules change.

## Tradeoffs ⇄

**Benefits:**
- Makes complex business logic visible and understandable to both technical and business stakeholders.
- Simplifies maintenance by separating business rules from application code.
- Enables completeness analysis: decision tables make it easy to spot missing rule combinations.
- Facilitates testing by providing a natural source for test case generation.

**Costs:**
- Not all business logic maps cleanly to a tabular format; some rules involve complex dependencies.
- Extracting rules from deeply embedded legacy code requires careful analysis and domain knowledge.
- Decision tables can become unwieldy if the number of conditions and combinations is very large.
- Introducing a rules engine adds a dependency and learning curve.

## Examples

A legacy insurance pricing system contains over 2,000 lines of nested conditional logic that calculates premiums based on age, location, coverage type, claim history, and policy tenure. No one fully understands all the interactions between conditions. The team extracts the pricing logic into decision tables, with one table per coverage type. Each row specifies a combination of input conditions and the resulting premium modifier. Business analysts review the tables and discover three condition combinations that produce incorrect pricing (the errors had been in production for years). The decision tables are then implemented using a lightweight rules engine, reducing the pricing code from 2,000 lines to 200 lines plus the externalized tables, and making future pricing changes a matter of editing a table rather than modifying code.
