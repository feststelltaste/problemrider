---
title: Subject Matter Reviews
description: Have work results reviewed and approved by domain experts
category:
- Process
- Communication
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/subject-matter-reviews
problems:
- misaligned-deliverables
- requirements-ambiguity
- stakeholder-developer-communication-gap
- implementation-rework
- inadequate-requirements-gathering
- poor-domain-model
- inconsistent-behavior
- quality-blind-spots
layout: solution
---

## How to Apply ◆

> In legacy modernization, subject matter reviews ensure that replacement implementations actually match the business intent behind legacy system behavior, not just its technical surface.

- Schedule regular review sessions where domain experts examine completed features against real business scenarios, not just acceptance criteria written by developers.
- Have domain experts walk through the replacement system using their actual daily workflows rather than scripted test cases — this reveals usability and correctness issues that formal testing misses.
- Include subject matter reviews at key milestones in the modernization, especially before decommissioning any legacy system component.
- Provide domain experts with side-by-side comparisons between legacy and replacement system outputs for the same inputs, making discrepancies immediately visible.
- Document domain expert feedback systematically and track resolution to build confidence that concerns are addressed rather than ignored.
- Select reviewers who represent different user groups and experience levels to capture diverse perspectives on the replacement system.

## Tradeoffs ⇄

> Subject matter reviews catch business-critical issues that technical reviews miss, but require access to busy domain experts and careful scheduling.

**Benefits:**

- Catches business logic errors that automated tests and code reviews miss because they require domain expertise to identify.
- Builds domain expert confidence in the modernization effort by giving them a voice in quality assurance.
- Surfaces undocumented business rules and edge cases that domain experts know intuitively but have never written down.
- Reduces the risk of deploying a technically correct but business-incorrect replacement.

**Costs and Risks:**

- Domain experts are often the busiest people in the organization, making it difficult to schedule regular review sessions.
- Reviews without clear structure can devolve into scope discussions or feature requests rather than quality validation.
- Domain experts may not understand the technical constraints that influenced design decisions, leading to feedback that is difficult to act on.
- Over-reliance on a small number of domain experts creates a knowledge bottleneck and single point of failure in the review process.

## How It Could Be

> The following scenario illustrates the value of subject matter reviews during legacy modernization.

A logistics company was replacing its freight rating engine, and automated tests showed 99.8% agreement with the legacy system's calculations. However, during a subject matter review, a senior rate analyst noticed that the replacement system was applying fuel surcharges before volumetric adjustments instead of after — a subtle ordering issue that the test data had not exposed because most test shipments were below the volumetric threshold. The analyst estimated that this error would have caused approximately $2 million in annual revenue leakage across high-volume shipping lanes. This single finding justified the entire investment in subject matter reviews for the project.
