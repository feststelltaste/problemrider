---
title: Stringly Typed Code
description: Domain concepts such as status, type, or identity are represented as
  raw, unvalidated strings instead of enums or value objects, pushing type checking
  from compile time to runtime.
category:
- Code
related_problems:
- slug: brittle-codebase
  similarity: 0.5
solutions:
- domain-driven-design
- value-range-definition
- static-analysis-and-linting
- typed-schema-extraction
- consistent-terminology
- code-conventions
- contract-testing
- property-based-testing
layout: problem
---

## Description

Stringly typed code represents domain concepts that have a small, well-defined set of valid values, or that carry business meaning, as plain strings rather than as enums, value objects, or other dedicated types. A user's status, a payment method, a permission level, or a workflow step end up passed around as `"ACTIVE"`, `"PAID"`, or `"admin"` instead of as a type the compiler can check. This shifts validation from compile time to runtime, or worse, to whichever function happens to parse the string first. Because there is no single place that owns the set of valid values, related checks, comparisons, and transformations are re-implemented wherever the string is used, and typos or unhandled values pass silently through the type system. The term is a play on "strongly typed" and has been used informally by developers for years to describe this specific flavor of primitive obsession.

## Indicators ⟡

- A single conceptual value, such as status, type, role, or mode, appears as a string literal in more than a handful of places, each with its own comparison logic.
- Validation for the same string value is implemented slightly differently in different modules.
- Code contains `if (status.equals("ACTIVE"))` or `switch` statements over string literals instead of enum constants.
- Typos in string literals, such as `"activ"` instead of `"active"`, are only caught, if at all, by a failing test or a bug report.
- The set of valid values for a field can only be discovered by grepping the codebase, not by reading a type definition.

## Symptoms ▲

- [Increased Risk of Bugs](increased-risk-of-bugs.md)
<br/>  Without a compiler-checked set of valid values, typos and unhandled cases in string comparisons slip through and surface as runtime defects.
- [Code Duplication](code-duplication.md)
<br/>  Because no single type owns the valid values or their validation, the same comparison and parsing logic gets re-implemented everywhere the string is used.
- [Dynamic Connascence](dynamic-connascence.md)
<br/>  Once a stringly typed value is also used to drive dynamic lookups, prefixes, or query paths, the value problem becomes a hidden runtime coupling between producer and consumer.
- [Difficult Code Comprehension](difficult-code-comprehension.md)
<br/>  Readers cannot discover the valid values or intended meaning of a field from its type; they must trace through the code to reconstruct an implicit enum.

## Causes ▼

- [Misunderstanding of OOP](misunderstanding-of-oop.md)
<br/>  Developers who don't reach for value objects or enums as a modeling tool default to the primitive type that happens to be convenient, which is usually a string.
- [Deadline Pressure](deadline-pressure.md)
<br/>  Introducing an enum or value object requires touching more places than passing a string along, so it is the shortcut under time pressure.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers unfamiliar with typed alternatives to raw strings don't recognize the long-term cost of skipping them.

## Detection Methods ○

- **String Literal Clustering:** Use static analysis to find the same string literal compared or assigned across many unrelated files, a sign it represents an unmodeled domain concept.
- **Equality Chain Search:** Grep for repeated `.equals("...")`, `== "..."`, or string-keyed switch statements against the same small set of values.
- **Typo Incident Review:** Check bug trackers for defects caused by a misspelled or unexpected string value that should have been an enum.
- **Schema Review:** Check database columns and API payload fields for free-text string types backing what is actually a fixed set of business states.

## Examples

An order management system represents order status as a `String` field that can be `"PENDING"`, `"SHIPPED"`, `"DELIVERED"`, or `"CANCELLED"`. Over time, five different modules each implement their own comparison logic: the shipping module checks `status.equals("SHIPPED")`, the reporting module checks `"shipped".equalsIgnoreCase(status)`, and the customer notification module checks a locally cached list of allowed strings that a developer forgot to update when `"RETURNED"` was introduced as a new status elsewhere. A customer's returned order silently fails to trigger a refund notification, because the notification module's hardcoded string list doesn't recognize the new status. If order status had been modeled as an enum, adding `RETURNED` would have forced the compiler to flag every `switch` statement that didn't handle the new case.
