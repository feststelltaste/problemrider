---
title: Architecture Conformity Analysis
description: Check the alignment of the software architecture with defined architectural
  principles
category:
- Architecture
problems:
- stagnant-architecture
- high-coupling-low-cohesion
- architectural-mismatch
- inconsistent-codebase
- ripple-effect-of-changes
- tight-coupling-issues
- monolithic-architecture-constraints
- technical-architecture-limitations
- circular-dependency-problems
layout: solution
related_solutions:
- slug: fitness-functions
  similarity: 0.8
- slug: architecture-reviews
  similarity: 0.8
- slug: static-analysis-and-linting
  similarity: 0.8
- slug: architecture-governance
  similarity: 0.75
- slug: architecture-decision-records
  similarity: 0.75
- slug: architecture-documentation
  similarity: 0.75
---

## Description

Architecture conformity analysis automatically checks whether a codebase's actual dependencies and communication patterns match a set of intended architectural rules — such as which layers may call which, or which modules may communicate directly — using tools like ArchUnit, Structure101, or Sonargraph that can encode these rules as executable checks run in continuous integration. In legacy systems the architecture that was originally designed and the architecture that actually exists in the code diverge steadily over years of urgent fixes and shortcuts, until the diagrams in whatever documentation still exists no longer describe how the system truly behaves, and nobody can say with confidence how many violations of the original design have accumulated. Making this drift visible is the necessary first step before it can be addressed: by encoding the intended rules as automated checks, previously invisible violations — such as presentation code calling database repositories directly, bypassing an entire business layer — become an explicit, countable metric rather than a vague sense that the architecture is not what it used to be. Because a legacy codebase with no history of enforcement will typically already contain a large number of violations, conformity analysis is usually introduced with a baseline of accepted violations and a target for gradual reduction, rather than a hard gate that would block all further development. Running these checks continuously also prevents new violations from being added while the team works through the existing backlog, which is what makes conformity analysis a durable process rather than a one-time cleanup, and it is often the discovery that legacy layer boundaries were fully enforceable after all that unlocks subsequent modernization steps such as swapping out a data access layer.

## How to Apply ◆

> In legacy systems, the actual architecture almost always diverges from the intended architecture — conformity analysis makes this drift visible and actionable.

- Define the intended architectural rules explicitly (layer dependencies, module boundaries, allowed communication patterns) in a format that can be checked automatically.
- Use architecture analysis tools (such as ArchUnit, Structure101, or Sonargraph) to detect violations of architectural rules in the existing codebase automatically.
- Run conformity checks as part of the continuous integration pipeline so that new violations are caught before they are merged.
- Start with the most critical architectural boundaries — such as the separation between domain logic and infrastructure — rather than trying to enforce all rules at once in a legacy codebase with many existing violations.
- Create a baseline of known violations and track reduction over time, treating conformity improvement as a measurable modernization goal.
- Review conformity analysis results in architecture review meetings to decide which violations to fix, which to accept temporarily, and which rules to revise.

## Tradeoffs ⇄

> Conformity analysis prevents architectural erosion but requires clear rules and team buy-in to be effective.

**Benefits:**

- Makes architectural violations visible before they accumulate into structural decay that is expensive to reverse.
- Provides objective, measurable criteria for architecture quality rather than relying on subjective assessments.
- Prevents new development from degrading the architecture while the team works on fixing existing violations.
- Supports gradual legacy modernization by tracking how architectural conformity improves over time.

**Costs and Risks:**

- Defining rules for a legacy system that has never had explicit architecture guidelines requires significant upfront effort and architectural judgment.
- Too many rules or overly strict rules can frustrate developers and lead to workarounds that circumvent the checks.
- Conformity analysis tools may require configuration effort and may not support all languages or frameworks used in legacy systems.
- Focusing solely on structural conformity can miss higher-level architectural issues like inappropriate technology choices or missing quality attributes.

## How It Could Be

> The following scenario shows how conformity analysis reveals and prevents architectural erosion.

A software company maintaining a 10-year-old enterprise application discovered through conformity analysis that its intended three-layer architecture (presentation, business, data access) had 340 violations where presentation layer classes directly accessed database repositories, bypassing the business layer entirely. These shortcuts had accumulated over years of urgent bug fixes and feature requests. The team configured ArchUnit rules to prevent new violations and established a "violation budget" that decreased by 10% each quarter. Over 18 months, the violation count dropped from 340 to 45, and the remaining violations were documented exceptions with explicit justification. The enforced layer boundaries made it possible to replace the data access layer with a new ORM without touching presentation code — a change that would have been impossible before the conformity effort.
