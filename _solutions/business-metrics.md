---
title: Business Metrics
description: Define business metrics to evaluate the functionality and quality of the software
category:
- Business
- Management
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/business-metrics
problems:
- declining-business-metrics
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- quality-blind-spots
- invisible-nature-of-technical-debt
- stakeholder-confidence-loss
layout: solution
---

## How to Apply ◆

- Identify key business outcomes the legacy system supports (revenue processing, customer onboarding time, order fulfillment rate) and define measurable metrics for each.
- Instrument the legacy system to collect these metrics, even if it requires adding lightweight monitoring code.
- Establish baselines for current metric values before beginning any modernization effort.
- Create dashboards that make business metrics visible to both technical and business stakeholders.
- Use business metrics to prioritize modernization work: focus on areas where poor system quality directly impacts business outcomes.
- Track metrics over time to demonstrate the value of modernization investments.

## Tradeoffs ⇄

**Benefits:**
- Provides objective evidence for investment decisions in legacy system improvement.
- Aligns technical work with business value, making it easier to secure stakeholder support.
- Reveals the true business impact of technical debt and legacy system limitations.
- Enables data-driven prioritization of modernization efforts.

**Costs:**
- Defining meaningful metrics requires close collaboration between business and technical teams.
- Instrumenting legacy systems for metric collection can be technically challenging.
- Poorly chosen metrics can incentivize the wrong behaviors or optimizations.
- Metric collection adds overhead to the system, though typically minimal.

## Examples

A legacy e-commerce platform suffers from slow page loads and frequent checkout failures, but the development team struggles to justify modernization investment because they cannot quantify the impact. They define business metrics: conversion rate, cart abandonment rate, average page load time, and revenue per session. After instrumenting the legacy system, they discover that checkout failures cost the business significant revenue monthly and that slow product page loads correlate with higher bounce rates. Armed with these numbers, the team secures funding for targeted performance improvements and can demonstrate measurable business improvement after each sprint of modernization work.
