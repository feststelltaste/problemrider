---
title: High Cohesion
description: Ensuring each module has a focused, well-defined purpose with closely
  related responsibilities
category:
- Architecture
- Code
problems:
- high-coupling-low-cohesion
- bloated-class
- god-object-anti-pattern
- monolithic-functions-and-classes
- spaghetti-code
- difficult-code-comprehension
- ripple-effect-of-changes
- tangled-cross-cutting-concerns
- excessive-class-size
- over-reliance-on-utility-classes
- poor-encapsulation
- circular-dependency-problems
- single-entry-point-design
layout: solution
related_solutions:
- slug: facades
  similarity: 0.75
- slug: dependency-injection
  similarity: 0.75
- slug: mediator
  similarity: 0.75
- slug: abstraction
  similarity: 0.75
- slug: layered-architecture
  similarity: 0.75
- slug: loose-coupling
  similarity: 0.75
---

## Description

High cohesion is the property that a module's contents — its methods, fields, and responsibilities — are closely related and serve a single, well-defined purpose, so that the module has one clear reason to change. It is diagnosed by looking for the opposite pattern: classes with vague names like "Manager" or "Helper," methods that operate on entirely disjoint subsets of a class's fields, or a single class accumulating unrelated concerns such as registration, billing, and scheduling all at once — the god object anti-pattern that is endemic to long-lived legacy codebases where new functionality was repeatedly added to whatever class was already convenient rather than to a purpose-built one. Restoring cohesion means identifying clusters of methods and data that genuinely belong together, based on which fields they actually use, and extracting each cluster into its own focused class or module, typically one responsibility at a time rather than in a single disruptive rewrite. This matters for legacy systems in particular because low cohesion is what makes small changes unpredictably risky: when unrelated responsibilities share mutable state inside one class, a change intended for one area can silently break another simply because they happen to live in the same file. The result of deliberately increasing cohesion is that responsibilities become independently understandable, testable, and changeable, which directly reduces the ripple effect of changes that plagues tightly entangled legacy code.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Analyze existing classes and modules for signs of low cohesion: multiple unrelated responsibilities, methods that do not use the same fields, or vague names like "Manager" or "Helper"
- Extract groups of related methods and data into new, focused classes or modules
- Use the Single Responsibility Principle as a guide: each module should have one reason to change
- Refactor god objects incrementally by moving one responsibility at a time into its own class
- Align module boundaries with domain concepts so that each module maps to a clear business capability
- Review method-to-field usage within classes to identify clusters that belong together

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Code becomes easier to understand because each module has a clear, limited purpose
- Changes are localized: modifying one responsibility does not risk breaking unrelated functionality
- Testing becomes simpler because focused modules have fewer dependencies and scenarios
- Improves team productivity by reducing cognitive load when working on individual components

**Costs and Risks:**
- Splitting modules increases the total number of files and may feel like over-engineering for small systems
- Requires careful identification of responsibility boundaries, which can be subjective
- Intermediate refactoring states may temporarily increase complexity before the full benefit is realized
- May surface hidden dependencies that were masked by the monolithic structure

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A healthcare application had a `PatientService` class with over 4,000 lines of code handling patient registration, billing, appointment scheduling, and medical record queries. Any change to billing logic risked breaking appointment scheduling because they shared mutable state within the class. The team systematically extracted each responsibility into its own service: `PatientRegistrationService`, `BillingService`, `AppointmentService`, and `MedicalRecordService`. Each new service was cohesive and independently testable. Bug rates in the patient module dropped noticeably in the following quarter, and developers reported spending far less time understanding code before making changes.
