---
title: User Acceptance Tests
description: Confirm fulfillment of requirements through formal acceptance tests with users
category:
- Testing
- Requirements
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/user-acceptance-tests
problems:
- misaligned-deliverables
- customer-dissatisfaction
- requirements-ambiguity
- insufficient-testing
- implementation-rework
- stakeholder-confidence-loss
- negative-user-feedback
- quality-blind-spots
layout: solution
---

## How to Apply ◆

> In legacy modernization, user acceptance tests serve as the final gate before decommissioning legacy components, ensuring the replacement actually works for the people who depend on it.

- Define acceptance criteria collaboratively with users before development begins, using concrete scenarios from their daily work with the legacy system.
- Structure UAT around complete business workflows rather than individual features — users need to verify that end-to-end processes work, not just isolated functions.
- Provide users with production-like data during UAT, ideally anonymized copies of real data from the legacy system, to ensure tests reflect actual usage conditions.
- Schedule UAT with enough time for users to perform thorough testing and for the development team to address findings before go-live deadlines.
- Track UAT defects separately from other defect types and require all critical UAT findings to be resolved before legacy system decommission approval.
- Include regression UAT cycles after significant changes to verify that fixes do not introduce new issues in previously accepted functionality.

## Tradeoffs ⇄

> UAT provides definitive validation that the replacement meets user needs but requires significant coordination and user commitment.

**Benefits:**

- Provides formal confirmation that the replacement system meets business requirements before the legacy system is retired, reducing go-live risk.
- Catches issues that automated tests and developer testing miss because they require real-world domain knowledge to identify.
- Creates accountability for sign-off, ensuring that users have explicitly approved the replacement before the legacy system is decommissioned.
- Builds user ownership of the replacement system by involving them in the quality assurance process.

**Costs and Risks:**

- UAT requires significant user time, which may conflict with their regular duties and lead to superficial testing under time pressure.
- If UAT is treated as a formality rather than genuine testing, critical issues will escape into production.
- Late-stage UAT discoveries can derail migration timelines if they reveal fundamental design issues that require extensive rework.
- Users may use UAT as an opportunity to request new features rather than validating agreed-upon requirements, leading to scope creep.

## How It Could Be

> The following scenario demonstrates the importance of structured UAT in legacy system replacement.

A wholesale distribution company was migrating from a legacy order management system to a modern platform. The development team had passed all automated tests and internal QA, but during UAT, order entry clerks discovered that the replacement system could not handle split shipments the way the legacy system did — the legacy system allowed clerks to split an order across warehouses during entry, while the new system required splitting after the order was submitted. This workflow difference would have added an extra step to every multi-warehouse order, affecting 30% of daily transactions. Because UAT was scheduled three weeks before the planned go-live, the team had time to implement the split-during-entry capability and conduct a regression UAT cycle. Without structured UAT, this issue would have been discovered on the first day of production use, potentially requiring a rollback to the legacy system.
