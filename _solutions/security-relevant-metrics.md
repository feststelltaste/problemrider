---
title: Security-Relevant Metrics
description: Define and collect metrics to quantify the security level
category:
- Security
- Management
problems:
- difficulty-quantifying-benefits
- invisible-nature-of-technical-debt
- monitoring-gaps
- quality-blind-spots
- poor-project-control
- modernization-roi-justification-failure
layout: solution
related_solutions:
- slug: security-metrics
  similarity: 0.95
- slug: security-frameworks
  similarity: 0.8
- slug: security-certification
  similarity: 0.8
- slug: threat-modeling
  similarity: 0.8
- slug: security-monitoring
  similarity: 0.8
- slug: security-training
  similarity: 0.75
---

## Description

Security-relevant metrics quantify a system's security posture through indicators such as vulnerability age, patch compliance rate, and attack surface size, turning security from a subjective impression into something that can be tracked, trended, and compared across a portfolio of applications. This is particularly valuable for organizations maintaining many legacy systems at once, where intuition about "which application is riskiest" is frequently wrong until the metrics are actually collected and compared side by side. Automating collection from vulnerability scanners, code analysis tools, and incident systems, and tracking leading indicators alongside lagging ones, gives teams an evidence base for prioritizing remediation investment rather than distributing it evenly or by whoever complains loudest — though metrics tied carelessly to incentives can be gamed, and some real risks resist quantification entirely.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A company maintaining a portfolio of legacy applications introduced a security metrics program tracking vulnerability density per application, mean time to remediate critical findings, and percentage of applications with current dependency versions. The metrics revealed that two of their 12 legacy applications accounted for 73% of all critical vulnerabilities and had remediation times three times longer than the portfolio average. This data enabled the security team to make a successful case for prioritized modernization of those two applications, resulting in a 50% reduction in portfolio-wide critical vulnerability count within one quarter.
