---
title: Inadequate Requirements Gathering
description: Insufficient analysis and documentation of requirements leads to building
  solutions that don't meet actual needs.
category:
- Process
- Testing
related_problems:
- slug: requirements-ambiguity
  similarity: 0.65
- slug: feedback-isolation
  similarity: 0.6
- slug: no-continuous-feedback-loop
  similarity: 0.6
- slug: poor-documentation
  similarity: 0.55
- slug: stakeholder-developer-communication-gap
  similarity: 0.55
- slug: feature-gaps
  similarity: 0.55
solutions:
- requirements-analysis
- stakeholder-feedback-loops
- evolutionary-requirements-development
- user-stories
- story-mapping
- on-site-customer
- personas
- specification-by-example
- behavior-driven-development-bdd
- business-process-modeling
- business-quality-scenarios
- subject-matter-reviews
- requirements-traceability-matrix
- acceptance-tests
- compatibility-requirements
- performance-budgets
- security-requirements-definition
layout: problem
---

## Description

Inadequate requirements gathering occurs when teams begin development without sufficiently understanding, analyzing, or documenting what needs to be built. This can involve rushing through requirements analysis, failing to engage the right stakeholders, missing edge cases, or not validating assumptions about user needs. Poor requirements gathering leads to solutions that don't address the actual problems, requiring costly rework and potentially failing to deliver business value.

## Indicators ⟡

- Development begins with vague or high-level requirements
- Key stakeholders are not involved in requirements definition
- Requirements documents are incomplete or ambiguous
- Edge cases and error conditions are not considered
- User workflows and business processes are not thoroughly understood

## Symptoms ▲

- [Implementation Rework](implementation-rework.md)
<br/>  Features must be rebuilt when initial understanding proves incorrect due to insufficient requirements analysis.
- [Misaligned Deliverables](misaligned-deliverables.md)
<br/>  Delivered features do not match stakeholder expectations because requirements were not properly understood upfront.
- [Feature Gaps](feature-gaps.md)
<br/>  Important functionality is missing because it was never identified during requirements gathering.
- [Scope Creep](scope-creep.md)
<br/>  Missing requirements are discovered during development, continuously expanding the project scope beyond original estimates.
- [Budget Overruns](budget-overruns.md)
<br/>  Rework and scope expansion from poor requirements drives costs beyond original budgets.
- [Assumption-Based Development](assumption-based-development.md)
<br/>  Developers make assumptions about what users need instead of validating requirements through proper analysis.
## Causes ▼

- [Stakeholder-Developer Communication Gap](stakeholder-developer-communication-gap.md)
<br/>  Poor communication between stakeholders and developers prevents effective requirements elicitation and validation.
- [Time Pressure](time-pressure.md)
<br/>  Pressure to start development quickly leads teams to rush through or skip thorough requirements analysis.
- [Feedback Isolation](feedback-isolation.md)
<br/>  Teams that do not regularly engage with stakeholders and users miss critical requirements and context.
## Detection Methods ○

- **Requirements Quality Assessment:** Evaluate completeness, clarity, and testability of requirements
- **Stakeholder Coverage Analysis:** Assess whether all relevant stakeholders contributed to requirements
- **Change Request Frequency:** Track how often requirements change during development
- **User Acceptance Testing Results:** Measure how well delivered solutions meet user expectations
- **Rework Percentage:** Calculate percentage of development effort spent on rework due to requirement issues

## Examples

A development team is asked to build a customer support ticketing system and receives high-level requirements like "track customer issues" and "assign tickets to support agents." Without deeper analysis, they build a basic system with ticket creation, assignment, and status updates. When they demo the system, support managers reveal they need complex routing rules based on customer tiers, integration with multiple communication channels, SLA tracking, escalation procedures, and reporting capabilities that weren't mentioned in the original requirements. The basic system they built cannot accommodate these needs and must be significantly redesigned. Another example involves an e-commerce team building a product recommendation engine based on the requirement to "show related products." They implement a simple algorithm based on product categories, but later discover the business actually needs personalized recommendations based on user behavior, purchase history, seasonal trends, and inventory levels. The simple category-based approach provides little business value and must be completely replaced with a more sophisticated system.
