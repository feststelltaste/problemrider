---
title: User Acceptance Tests
description: Confirm fulfillment of requirements through formal acceptance tests with
  users
category:
- Testing
- Requirements
problems:
- misaligned-deliverables
- customer-dissatisfaction
- requirements-ambiguity
- insufficient-testing
- implementation-rework
- stakeholder-confidence-loss
- negative-user-feedback
- quality-blind-spots
- reduced-feature-quality
layout: solution
related_solutions:
- slug: usability-tests
  similarity: 0.75
- slug: acceptance-tests
  similarity: 0.75
- slug: user-stories
  similarity: 0.75
- slug: prototypes
  similarity: 0.7
- slug: requirements-analysis
  similarity: 0.7
- slug: user-centered-design
  similarity: 0.7
---

## Description

User acceptance testing is a formal validation stage in which the actual users of a system — not developers, not QA engineers — verify that a replacement or new capability correctly supports the real business workflows they depend on, using acceptance criteria agreed on collaboratively before development began. It differs from automated and developer-led testing in what it can catch: internal correctness and unit-level behavior are covered elsewhere, but only the people who perform the work daily can recognize when a technically correct implementation nonetheless fails to match how the job actually gets done. This distinction is decisive in legacy modernization, where replacement systems are built against documented requirements that inevitably miss tacit knowledge embedded in years of undocumented workarounds and habitual usage patterns that never made it into any specification. Structuring UAT around complete end-to-end business workflows, run against production-like data, surfaces exactly these gaps — a legacy capability quietly relied upon that the new system's designers never knew to replicate — while there is still time before the legacy system is decommissioned and rollback becomes costly or impossible. Because UAT sits at the very end of the delivery pipeline, findings at this stage can be schedule-threatening, which makes explicit sign-off criteria and adequate lead time before go-live essential rather than optional.

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
