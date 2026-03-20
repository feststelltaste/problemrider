---
title: Specification by Example
description: Collaboratively defining requirements through concrete examples that become executable specifications
category:
- Requirements
- Testing
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/specification-by-example
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- misaligned-deliverables
- implementation-rework
- insufficient-testing
- stakeholder-developer-communication-gap
- legacy-code-without-tests
- inconsistent-behavior
layout: solution
---

## How to Apply ◆

> In legacy modernization, specification by example bridges the gap between undocumented legacy behavior and clearly defined replacement requirements by using concrete examples as the shared language between domain experts and developers.

- Conduct collaborative specification workshops where developers, testers, and domain experts work together to define expected system behavior through concrete input-output examples drawn from real legacy system usage.
- Use the legacy system itself to generate examples — run representative scenarios through the old system and record the results as the initial specification for the replacement.
- Express examples in a structured format (such as Given-When-Then) that can be automated as executable tests, ensuring that specifications remain verifiable throughout the modernization.
- Focus examples on business rules and edge cases where legacy behavior is most complex or least documented, since these are the areas most likely to cause defects during replacement.
- Automate the examples as acceptance tests that run against both the legacy system (to verify correctness) and the new system (to verify parity), providing a clear measure of migration progress.
- Maintain a living documentation repository where examples are organized by business capability, serving as both specification and test suite.

## Tradeoffs ⇄

> Specification by example creates alignment and living documentation but requires sustained collaboration between technical and business stakeholders.

**Benefits:**

- Eliminates ambiguity in requirements by replacing abstract descriptions with concrete, verifiable examples that everyone can understand.
- Creates executable tests as a byproduct of the specification process, ensuring that the replacement system behaves correctly from the start.
- Preserves critical business knowledge that exists only in the legacy system's behavior, capturing it in a format that outlives the old system.
- Provides a clear, measurable definition of "done" for each migrated feature — the examples either pass or they do not.

**Costs and Risks:**

- Requires regular access to domain experts who understand legacy system behavior, which may be difficult to secure.
- The workshop format can be time-consuming, especially when specifying complex legacy behavior with many edge cases.
- Examples that are too detailed can become brittle tests that break with minor implementation changes.
- Teams may struggle to find the right level of abstraction — too few examples miss critical edge cases, while too many become unmanageable.

## Examples

> The following scenario demonstrates specification by example during a legacy system migration.

A payroll processing company was replacing its legacy system that handled tax calculations for 12 different jurisdictions. Rather than attempting to write traditional requirements documents for the thousands of tax rules, the team held weekly specification workshops with payroll tax specialists. In each session, the specialists provided concrete payroll scenarios — specific employees, specific pay periods, specific deduction combinations — and walked through the expected calculations step by step. These examples were automated as executable specifications that ran against both the legacy system and the new implementation. When the specifications produced different results between systems, the team investigated whether the discrepancy was a legacy bug or a migration defect. Over eight months, the team accumulated 2,400 executable examples that served as both the specification and the regression test suite for the entire migration.
