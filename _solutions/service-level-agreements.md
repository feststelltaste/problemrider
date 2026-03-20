---
title: Service Level Agreements
description: Defining expectations for software availability and performance
category:
- Management
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/service-level-agreements
problems:
- poor-operational-concept
- stakeholder-frustration
- unclear-goals-and-priorities
- constant-firefighting
- customer-dissatisfaction
- system-outages
- modernization-roi-justification-failure
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Define measurable availability and performance targets for legacy system services based on business needs
- Specify SLA metrics clearly: uptime percentage, response time percentiles, error rate thresholds
- Establish measurement methodology and reporting frequency for each SLA metric
- Define consequences and remediation processes when SLAs are not met
- Align SLA targets with the realistic capabilities of the legacy system and planned improvement trajectory
- Use SLA data to justify investment in legacy system modernization and reliability improvements
- Review and adjust SLAs periodically as system capabilities and business requirements evolve

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Creates shared expectations between business stakeholders and engineering teams
- Provides objective criteria for evaluating legacy system performance
- Justifies investment in reliability improvements with measurable targets
- Enables data-driven prioritization of maintenance versus feature work

**Costs and Risks:**
- Setting SLAs too aggressively for legacy systems creates perpetual failure and team demoralization
- Measuring SLAs requires monitoring infrastructure that legacy systems may lack
- SLAs can become politicized and used as blame instruments rather than improvement tools
- Narrow SLA focus may neglect important quality dimensions not covered by the agreement

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A B2B SaaS company operated a legacy platform with no defined availability commitments. Customers complained about frequent outages, but without SLAs the engineering team had no framework for prioritizing reliability work against feature requests. After establishing a 99.5% monthly availability SLA with defined measurement methodology, the team discovered they were currently achieving only 98.2%. This data justified a dedicated reliability improvement initiative that brought availability above the SLA target within three months. The explicit target also helped the team push back on feature requests that would have compromised stability.
