---
title: Security-Relevant Metrics
description: Define and collect metrics to quantify the security level
category:
- Security
- Management
quality_tactics_url: https://qualitytactics.de/en/security/security-relevant-metrics
problems:
- difficulty-quantifying-benefits
- invisible-nature-of-technical-debt
- monitoring-gaps
- quality-blind-spots
- poor-project-control
- modernization-roi-justification-failure
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify key security indicators relevant to the legacy system such as vulnerability age, patch compliance, and attack surface size
- Automate collection of metrics from vulnerability scanners, code analysis tools, and incident management systems
- Track leading indicators (e.g., security training completion, patch cadence) alongside lagging indicators (e.g., incident count, breach impact)
- Benchmark metrics against industry standards and historical baselines to contextualize results
- Present metrics in formats appropriate for different audiences: technical detail for teams, trends for management
- Use metrics to set measurable security improvement goals and track progress quarterly

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Transforms security from a subjective assessment into a measurable capability
- Supports business case development for security investments in legacy systems
- Enables trend analysis that reveals whether security posture is improving or degrading
- Provides early warning signals before security issues become incidents

**Costs and Risks:**
- Collecting meaningful metrics from legacy systems with limited instrumentation can be challenging
- Metrics can be gamed if they are tied to incentives without proper design
- Over-reliance on metrics can create blind spots for risks that are not easily quantified
- Metric programs require ongoing curation to remain relevant as the threat landscape evolves

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A company maintaining a portfolio of legacy applications introduced a security metrics program tracking vulnerability density per application, mean time to remediate critical findings, and percentage of applications with current dependency versions. The metrics revealed that two of their 12 legacy applications accounted for 73% of all critical vulnerabilities and had remediation times three times longer than the portfolio average. This data enabled the security team to make a successful case for prioritized modernization of those two applications, resulting in a 50% reduction in portfolio-wide critical vulnerability count within one quarter.
