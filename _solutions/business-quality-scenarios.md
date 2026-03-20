---
title: Business Quality Scenarios
description: Specify and verify quality requirements through business-driven scenarios
category:
- Requirements
- Testing
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/business-quality-scenarios
problems:
- requirements-ambiguity
- quality-blind-spots
- inadequate-requirements-gathering
- difficulty-quantifying-benefits
- stakeholder-developer-communication-gap
layout: solution
---

## How to Apply ◆

- Define quality scenarios using the stimulus-response format: who/what triggers the scenario, what happens, and what measurable response is expected.
- Derive scenarios from real business concerns (e.g., "When 500 users submit orders simultaneously during a sale event, 99% of orders must complete within 3 seconds").
- Prioritize scenarios based on business impact and use them to guide architectural decisions in legacy modernization.
- Automate verification of quality scenarios where possible, integrating them into performance and integration test suites.
- Review and update quality scenarios as business requirements evolve.
- Use quality scenarios to communicate non-functional requirements in terms business stakeholders understand.

## Tradeoffs ⇄

**Benefits:**
- Translates abstract quality requirements into concrete, testable, and business-relevant scenarios.
- Provides clear acceptance criteria for non-functional requirements that are often left vague.
- Helps prioritize architectural investments by tying quality attributes to business value.

**Costs:**
- Defining meaningful quality scenarios requires collaboration between business and technical teams.
- Not all quality attributes are easy to express as business scenarios.
- Automated verification of quality scenarios may require specialized testing infrastructure.
- Scenarios need regular review to stay aligned with evolving business needs.

## Examples

A legacy banking application must meet strict availability and performance requirements, but these are expressed only as vague statements like "the system should be fast and reliable." The team works with business stakeholders to define concrete quality scenarios: "During month-end processing, when 200 concurrent users run balance reports, each report must complete within 5 seconds" and "If the primary database fails, the system must failover to the standby within 30 seconds with no data loss." These scenarios guide the modernization effort by making it clear which quality improvements deliver business value and which are merely technical preferences. The team builds automated tests that verify these scenarios in staging environments before each release.
