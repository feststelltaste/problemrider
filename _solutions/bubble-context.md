---
title: Bubble Context
description: Clearly distinguish extensions from existing code parts
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/maintainability/bubble-context
problems:
- legacy-business-logic-extraction-difficulty
- high-coupling-low-cohesion
- spaghetti-code
- fear-of-change
- brittle-codebase
- monolithic-architecture-constraints
- inconsistent-codebase
layout: solution
---

## How to Apply ◆

> In legacy systems, bubble context creates a clean boundary around new code, preventing it from being contaminated by legacy patterns while coexisting with the old system.

- Define a clear boundary (the "bubble") around new functionality, with explicit translation layers at every point where new code must interact with legacy code.
- Implement new features using modern practices and patterns inside the bubble, without being constrained by the legacy system's coding conventions, data models, or architectural patterns.
- Create adapter classes or modules at the bubble boundary that translate between the legacy system's data formats and the bubble's internal model.
- Use the bubble context pattern when adding new features to a legacy system that cannot yet be fully decomposed — it allows incremental improvement without attempting a complete rewrite.
- Keep the bubble's internal model clean and domain-driven, even if the legacy system's model is denormalized, inconsistent, or poorly named.
- Test the bubble independently from the legacy system using the adapters as seams for test doubles, ensuring that new code can be verified without legacy system dependencies.

## Tradeoffs ⇄

> Bubble context enables clean new development within legacy constraints but creates a dual-model system that must be managed.

**Benefits:**

- Allows new features to be built with modern practices without waiting for the entire legacy system to be modernized.
- Prevents "legacy contamination" where new code adopts the legacy system's poor patterns simply because it needs to interact with them.
- Creates natural migration boundaries — the bubble can eventually grow to replace the legacy system as more functionality moves inside it.
- Enables the team to develop and test new features faster by isolating them from legacy complexity.

**Costs and Risks:**

- The translation layer at the bubble boundary adds complexity and must be maintained as both the legacy system and the bubble evolve.
- Multiple bubbles within the same legacy system can create a patchwork architecture that is harder to understand than either a pure legacy or pure modern system.
- Teams may disagree on where bubble boundaries should be drawn, leading to inconsistent application of the pattern.
- The bubble's clean model and the legacy model can drift apart, making the translation layer increasingly complex over time.

## How It Could Be

> The following scenario illustrates how bubble context enables clean development within a legacy codebase.

An energy company needed to add a new real-time pricing feature to its legacy billing system. The legacy system used a flat relational model with 200-character column names following a 1990s naming convention, and all business logic was embedded in stored procedures. Rather than building the pricing feature in the same style, the team created a bubble context with a clean domain model (using classes like `PricingPlan`, `RateSchedule`, and `ConsumptionTier`) and modern code patterns. Adapters at the boundary translated between the legacy database's `CUST_RATE_SCHED_EFF_DT` format and the bubble's `RateSchedule.effectiveDate`. The pricing feature was developed and tested in half the time it would have taken using legacy patterns, and the bubble's clean architecture later served as the foundation for replacing additional billing modules.
