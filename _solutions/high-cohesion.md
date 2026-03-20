---
title: High Cohesion
description: Ensuring each module has a focused, well-defined purpose with closely related responsibilities
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/maintainability/high-cohesion
problems:
- high-coupling-low-cohesion
- bloated-class
- god-object-anti-pattern
- monolithic-functions-and-classes
- spaghetti-code
- difficult-code-comprehension
- ripple-effect-of-changes
- tangled-cross-cutting-concerns
layout: solution
---

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
