---
title: Data Strategy
description: Define common data standards, formats, and integration patterns across systems
category:
- Architecture
- Management
quality_tactics_url: https://qualitytactics.de/en/compatibility/data-strategy
problems:
- cross-system-data-synchronization-problems
- poor-domain-model
- system-integration-blindness
- integration-difficulties
- data-migration-complexities
- technology-stack-fragmentation
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Define an organization-wide data strategy covering data ownership, quality standards, and integration patterns
- Establish canonical data models for core business entities shared across systems
- Choose and standardize integration patterns (event-driven, API-based, batch) for different use cases
- Assign data stewards responsible for the quality and evolution of key data domains
- Create a data integration roadmap that prioritizes consolidation of the most problematic legacy data flows
- Review and update the data strategy periodically to reflect changes in the system landscape

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides a coherent vision for how data flows across the organization, reducing ad hoc integration
- Enables informed decisions about data format and storage choices during legacy modernization
- Reduces data quality issues caused by inconsistent standards across systems

**Costs and Risks:**
- Developing a comprehensive data strategy requires cross-functional alignment and executive sponsorship
- Strategy without execution becomes shelfware that does not improve the legacy landscape
- Centralized data governance may conflict with team autonomy in decentralized organizations
- Keeping the strategy current requires ongoing investment

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An insurance company with 20 legacy systems had no data strategy, resulting in customer data spread across seven different formats and five databases with no single source of truth. Claims adjusters spent an average of 30 minutes per claim reconciling customer information. After defining a data strategy with canonical models, assigned data stewards, and an event-driven integration pattern for customer data, the company achieved a unified customer view within 14 months. Claims processing time dropped by 25%.
