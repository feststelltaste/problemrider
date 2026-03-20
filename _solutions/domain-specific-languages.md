---
title: Domain-Specific Languages
description: Use programming languages specifically adapted to the domain for business expressions and rules
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/domain-specific-languages
problems:
- complex-and-obscure-logic
- legacy-business-logic-extraction-difficulty
- difficult-code-comprehension
- poor-domain-model
- stakeholder-developer-communication-gap
- requirements-ambiguity
layout: solution
---

## How to Apply ◆

- Identify areas of the legacy codebase where business rules are expressed in general-purpose code that is difficult for non-developers to understand.
- Design a domain-specific language (DSL) that allows business rules to be expressed in terms natural to the domain (e.g., pricing rules, validation rules, workflow definitions).
- Implement the DSL using an internal approach (fluent API within the existing programming language) or an external approach (custom syntax with a parser).
- Migrate business rules from the legacy code into the DSL incrementally, validating each migration against the existing behavior.
- Enable domain experts to read and review rules expressed in the DSL, even if they cannot author them directly.
- Provide tooling support: syntax highlighting, validation, and testing capabilities for the DSL.

## Tradeoffs ⇄

**Benefits:**
- Makes business rules readable by domain experts, enabling direct validation of implementation correctness.
- Separates business rule expression from technical implementation concerns.
- Reduces the effort to modify business rules when they change, as changes are expressed in domain terms.
- Can significantly reduce the volume of code needed to express complex business logic.

**Costs:**
- Designing and implementing a DSL requires specialized skills and significant upfront investment.
- Developers must learn the DSL in addition to the general-purpose language.
- Poorly designed DSLs can be harder to understand than the general-purpose code they replace.
- DSLs require maintenance: the language itself, its tooling, and its documentation.
- Debugging DSL-expressed logic can be challenging if error messages map poorly to the domain language.

## Examples

A legacy insurance company has premium calculation rules embedded in thousands of lines of Java code with deeply nested conditionals. Business analysts cannot verify whether the code correctly implements their pricing models. The team creates an internal DSL using a fluent API that reads like natural language: `when(driver.age().isBelow(25)).and(vehicle.type().is("sports")).then(applyMultiplier(1.8))`. The existing Java rules are migrated to DSL expressions one by one, with tests verifying equivalent behavior. Actuaries can now read the pricing rules directly and spot errors. When a regulatory change requires new pricing factors, the change is expressed in the DSL and implemented in hours rather than the weeks it previously took to modify and test the legacy Java code.
