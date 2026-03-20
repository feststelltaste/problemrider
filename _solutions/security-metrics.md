---
title: Security Metrics
description: Define, collect, and evaluate metrics to quantify the security status
category:
- Security
- Management
quality_tactics_url: https://qualitytactics.de/en/security/security-metrics
problems:
- difficulty-quantifying-benefits
- invisible-nature-of-technical-debt
- monitoring-gaps
- quality-blind-spots
- insufficient-audit-logging
- poor-project-control
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Define meaningful security metrics aligned with organizational risk appetite and security objectives
- Collect metrics such as mean time to patch, vulnerability density, incident frequency, and false positive rates
- Automate metric collection through integration with security tools, issue trackers, and monitoring systems
- Create dashboards that present security metrics to both technical teams and executive stakeholders
- Establish baselines and set improvement targets for key security indicators
- Review metrics regularly in security governance meetings and use them to drive resource allocation decisions
- Track trends over time rather than focusing on individual data points

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables data-driven security decisions and investment prioritization
- Makes security posture visible and communicable to non-technical stakeholders
- Identifies trends that signal improving or degrading security before incidents occur
- Supports accountability by making security performance measurable

**Costs and Risks:**
- Poorly chosen metrics can incentivize counterproductive behavior (e.g., closing findings without fixing them)
- Metric collection adds overhead to already burdened legacy system teams
- Security metrics can create false confidence if they measure activity rather than effectiveness
- Legacy systems may lack the instrumentation needed for automated metric collection

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A technology company struggling to justify its security budget for legacy system maintenance began tracking four key metrics: average days to patch critical vulnerabilities, number of open high-severity findings, percentage of code covered by security tests, and mean time to detect security incidents. After six months of tracking, the data showed that their legacy systems had a mean patch time of 67 days compared to 12 days for newer systems, providing concrete justification for a dedicated legacy security improvement initiative. The metrics dashboard also revealed that 80% of their open findings were concentrated in two legacy components, enabling focused remediation.
