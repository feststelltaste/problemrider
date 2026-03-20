---
title: Behavior-Driven Development (BDD)
description: Development based on expected system behaviors
category:
- Testing
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/behavior-driven-development-bdd
problems:
- requirements-ambiguity
- insufficient-testing
- legacy-code-without-tests
- stakeholder-developer-communication-gap
- misaligned-deliverables
- poor-test-coverage
- implementation-rework
layout: solution
---

## How to Apply ◆

> In legacy modernization, BDD creates a shared specification language that captures legacy system behavior in a format both domain experts and developers can verify.

- Use Given-When-Then scenarios to document the legacy system's current behavior before modifying it, creating executable specifications that serve as both documentation and regression tests.
- Conduct "three amigos" sessions (developer, tester, domain expert) to write BDD scenarios for each legacy feature being migrated, capturing edge cases that only domain experts know about.
- Choose a BDD framework appropriate for the legacy system's technology stack (Cucumber, SpecFlow, Behave) and integrate it into the continuous integration pipeline.
- Write scenarios at the business behavior level rather than the UI or technical level, so they remain valid even when the underlying implementation changes during modernization.
- Use BDD scenarios as acceptance criteria for migration stories — a feature is considered successfully migrated when all its BDD scenarios pass against the new implementation.
- Build a scenario library organized by business capability to create a living documentation system that replaces outdated specification documents.

## Tradeoffs ⇄

> BDD creates living documentation and aligns teams around behavior but requires consistent participation from domain experts.

**Benefits:**

- Creates executable specifications that serve as both tests and documentation, solving the problem of specifications that drift from implementation.
- Bridges the communication gap between technical and business stakeholders by using a shared language that both can read and validate.
- Provides a clear migration completion metric — the percentage of BDD scenarios passing against the new system.
- Catches behavioral regressions during modernization that unit tests might miss because they test implementation rather than behavior.

**Costs and Risks:**

- BDD scenarios require ongoing access to domain experts, who may not be available for the sustained engagement needed.
- Poorly written scenarios that are too detailed or too technical lose their value as a communication tool and become just another test format.
- The step definition layer between scenarios and code can become a maintenance burden if not kept clean and well-organized.
- Teams may focus on writing scenarios for easy cases and avoid the complex edge cases where BDD provides the most value.

## How It Could Be

> The following scenario shows how BDD supports legacy system migration.

A logistics company migrating its shipment tracking system used BDD to capture the complex business rules around delivery time window calculations. The legacy system computed delivery windows differently based on carrier, destination zone, package weight, and service level — rules that existed only in procedural code and the memory of two senior developers. Through structured BDD workshops, the team wrote 180 Given-When-Then scenarios covering all combinations. When the first implementation of the new calculation engine was tested against these scenarios, 23 failed — revealing edge cases where the new implementation diverged from legacy behavior. Twelve of these were genuine bugs in the new code, and eleven turned out to be bugs in the legacy system that the business decided to fix rather than replicate. The BDD scenarios became the authoritative specification for delivery window calculations, outliving both the legacy system and the original developers' tenure.
