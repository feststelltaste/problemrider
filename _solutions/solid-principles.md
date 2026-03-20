---
title: SOLID Principles
description: Apply fundamental design principles for object-oriented programming
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/solid-principles/
problems:
- high-coupling-low-cohesion
- circular-references
- hidden-side-effects
- ripple-effect-of-changes
- misunderstanding-of-oop
- procedural-background
- procedural-programming-in-oop-languages
- insufficient-design-skills
- single-entry-point-design
- convenience-driven-development
- defensive-coding-practices
- uncontrolled-codebase-growth
- increased-technical-shortcuts
layout: solution
---

## How to Apply ◆

> In legacy systems, SOLID principles are not an academic exercise — they are the most direct path to reducing the pain of every change. Introducing them incrementally, starting where the pain is worst, transforms a codebase that resists change into one that accommodates it.

- Start with the Single Responsibility Principle (SRP) in the areas with the highest churn: identify the god classes and single-entry-point controllers that accumulate responsibilities over years of feature additions. Extract one responsibility at a time into a new class, keeping the original class as a thin coordinator until it shrinks to a manageable size.
- Apply the Open/Closed Principle (OCP) when you find yourself editing the same switch statement or if-else chain every time a new variant is added. Replace conditional branching with polymorphism by introducing an interface and one implementation per variant, so new variants can be added without modifying existing code.
- Use the Liskov Substitution Principle (LSP) to audit existing inheritance hierarchies in the legacy codebase. Look for subclasses that override methods by throwing exceptions or silently ignoring parameters — these are LSP violations that create hidden side effects and unpredictable behavior. Replace broken hierarchies with composition or properly designed interfaces.
- Enforce the Interface Segregation Principle (ISP) by splitting large interfaces that force implementors to provide stub implementations for methods they do not need. In legacy systems, these bloated interfaces are often the reason why components become tightly coupled to contracts they only partially fulfill.
- Introduce the Dependency Inversion Principle (DIP) by replacing direct instantiation of dependencies with constructor injection. In legacy code, this often means wrapping concrete dependencies behind interfaces so that high-level business logic no longer depends on low-level infrastructure details like specific database drivers or email libraries.
- Teach SOLID principles through code review, not through lecture. When reviewing legacy code modifications, point out specific violations and suggest concrete refactoring steps. This builds understanding in the context of the team's actual codebase rather than in the abstract.
- Address procedural-style code in OOP languages by demonstrating how SOLID principles naturally lead to objects that encapsulate both state and behavior, replacing the pattern of utility classes operating on passive data structures.
- Use static analysis rules to detect common SOLID violations automatically — classes with too many dependencies (SRP), methods that modify unrelated state (SRP), and concrete class references where interfaces should be used (DIP).

## Tradeoffs ⇄

> SOLID principles provide a shared design vocabulary that makes legacy code incrementally more maintainable, but they require judgment about when and how strictly to apply each principle.

**Benefits:**

- Reduces coupling by ensuring each class has a single reason to change, which directly shrinks the blast radius of modifications and prevents the ripple effect that makes legacy changes so expensive.
- Makes code more predictable by eliminating hidden side effects: when each class has a clear, focused responsibility, developers can understand what a function does from its name and signature without tracing through unrelated behaviors.
- Provides developers with procedural backgrounds a structured path to object-oriented thinking by offering concrete rules rather than abstract OOP philosophy.
- Improves testability because classes that depend on abstractions rather than concrete implementations can be tested in isolation with simple test doubles.
- Creates a common design language for code review discussions, replacing subjective "this feels wrong" feedback with specific, actionable principle references.

**Costs and Risks:**

- Over-application of SOLID principles creates an explosion of small classes and interfaces that can be as hard to navigate as the original monolithic code — the result is indirection without clarity.
- Introducing interfaces and dependency injection into a legacy codebase that has no IoC container requires infrastructure work before the design benefits materialize, and this work competes with feature delivery.
- Developers unfamiliar with SOLID may produce worse designs by mechanically applying the principles without understanding their intent — for example, creating one-method interfaces for every class regardless of whether multiple implementations will ever exist.
- Refactoring legacy code toward SOLID principles without adequate test coverage carries the same risk as any refactoring: behavior may change in ways that are not detected until production.
- In some legacy contexts, the procedural style is actually appropriate — batch processing scripts, data migration utilities, and simple CRUD operations may not benefit from full SOLID treatment, and forcing OOP patterns onto them adds complexity without value.

## Examples

> The following scenarios illustrate how SOLID principles address concrete design problems encountered in legacy systems.

A logistics company maintained an order processing system where a single `OrderService` class had grown to 3,200 lines over eight years. It validated orders, calculated shipping costs, applied discount rules, updated inventory, sent email notifications, and logged audit trails. Every new shipping carrier or discount promotion required modifying this class, and changes to email formatting occasionally broke discount calculations due to shared instance variables. The team applied SRP by extracting each responsibility into its own class — `ShippingCalculator`, `DiscountEngine`, `InventoryUpdater`, `NotificationSender`, and `AuditLogger` — each with a clearly defined interface. The `OrderService` was reduced to a 60-line orchestrator. When the company added a new shipping carrier the following quarter, the change required adding one new class implementing the `ShippingCalculator` interface and zero modifications to existing classes, following OCP.

A banking application had a `ReportGenerator` class that was subclassed for different report types, but several subclasses threw `UnsupportedOperationException` for methods they inherited but could not meaningfully implement. When a batch job iterated over all report generators and called `generateSummary()`, some subclasses silently failed, producing incomplete reports that were only caught by end-of-month reconciliation. The team recognized these as LSP violations and restructured the hierarchy into separate interfaces — `DetailedReport` and `SummaryReport` — so each implementation only committed to capabilities it could actually deliver. The batch job was updated to check interface types, and the silent failures were eliminated entirely.

A healthcare software company hired developers with strong C and COBOL backgrounds to maintain a Java-based patient management system. The codebase was dominated by static utility classes — `PatientUtils`, `BillingUtils`, `ScheduleUtils` — each containing dozens of static methods that operated on plain data objects. Adding new patient types required modifying every utility class. The team introduced SOLID principles through a series of pair programming sessions where they demonstrated how DIP and ISP could replace the utility-class pattern with proper domain objects that encapsulated behavior. Over six months, the team converted the most frequently modified utilities into service classes with injected dependencies, reducing the average number of files changed per feature from fourteen to three.
