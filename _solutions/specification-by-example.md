---
title: Specification by Example
description: Collaboratively defining requirements through concrete examples that
  become executable specifications
category:
- Requirements
- Testing
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- misaligned-deliverables
- implementation-rework
- insufficient-testing
- stakeholder-developer-communication-gap
- legacy-code-without-tests
- inconsistent-behavior
- reduced-feature-quality
- frequent-changes-to-requirements
layout: solution
related_solutions:
- slug: behavior-driven-development-bdd
  similarity: 0.85
- slug: evolutionary-requirements-development
  similarity: 0.75
- slug: user-stories
  similarity: 0.75
- slug: acceptance-tests
  similarity: 0.75
- slug: requirements-analysis
  similarity: 0.75
- slug: living-documentation
  similarity: 0.7
---

## Description

Specification by example is a collaborative practice in which developers, testers, and domain experts define expected system behavior through concrete, real-world input-output pairs rather than abstract prose requirements, and then express those examples in a structured, automatable format so they double as executable tests. The technique directly addresses one of the hardest problems in legacy modernization: the original requirements were often never documented, the people who wrote the system are gone, and the only remaining source of truth for what the system is actually supposed to do is its own observed behavior — which the legacy system itself can be run against to generate the initial set of examples. Because each example is concrete rather than abstract, it eliminates the ambiguity that causes replacement systems to diverge subtly from legacy behavior on edge cases nobody thought to write down, and because the examples are automated, they can be run against both the old and new systems simultaneously to produce a direct, continuously verifiable measure of migration parity. The resulting body of examples becomes living documentation that outlives the legacy system itself, capturing business rules that existed only as implicit behavior. The practice does require regular, sustained access to domain experts who understand the legacy system's quirks, and finding the right granularity of examples — enough to cover meaningful edge cases without becoming an unmanageable, brittle mass — is a skill that takes iteration to develop.

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

## How It Could Be

> The following scenario demonstrates specification by example during a legacy system migration.

A payroll processing company was replacing its legacy system that handled tax calculations for 12 different jurisdictions. Rather than attempting to write traditional requirements documents for the thousands of tax rules, the team held weekly specification workshops with payroll tax specialists. In each session, the specialists provided concrete payroll scenarios — specific employees, specific pay periods, specific deduction combinations — and walked through the expected calculations step by step. These examples were automated as executable specifications that ran against both the legacy system and the new implementation. When the specifications produced different results between systems, the team investigated whether the discrepancy was a legacy bug or a migration defect. Over eight months, the team accumulated 2,400 executable examples that served as both the specification and the regression test suite for the entire migration.
