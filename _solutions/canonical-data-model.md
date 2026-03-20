---
title: Canonical Data Model
description: Standardizing a shared data model across systems instead of point-to-point transformations
category:
- Architecture
- Database
quality_tactics_url: https://qualitytactics.de/en/compatibility/canonical-data-model
problems:
- cross-system-data-synchronization-problems
- integration-difficulties
- poor-interfaces-between-applications
- data-migration-complexities
- inconsistent-behavior
- poor-domain-model
- technology-stack-fragmentation
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Define a canonical data model that represents the shared business concepts used across systems
- Build translators at the boundary of each system that map between the system's internal model and the canonical model
- Start with the highest-traffic or most error-prone integration points rather than modeling everything at once
- Version the canonical model and manage its evolution through a governance process
- Store the canonical schema in a shared repository accessible to all teams
- Use the canonical model as the contract for event-driven or messaging-based integrations

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces the number of integration mappings from O(n^2) point-to-point to O(n) translations
- Creates a shared vocabulary that improves cross-team communication about data
- Simplifies adding new systems to the integration landscape

**Costs and Risks:**
- Designing the canonical model requires cross-team consensus, which can be slow and political
- The canonical model can become a lowest-common-denominator that loses important domain nuances
- Changes to the canonical model ripple across all connected systems
- Risk of creating an overly generic model that serves no system well

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A mid-size bank had 12 legacy systems exchanging customer data through 40 point-to-point integrations, each with its own field mapping logic. Data inconsistencies between systems caused an average of five reconciliation incidents per month. The team defined a canonical customer model and built translators for each system over six months. The number of integration mappings dropped from 40 to 12, reconciliation incidents fell to fewer than one per month, and onboarding a new CRM system took three weeks instead of the previously estimated three months.
