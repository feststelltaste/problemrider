---
title: Root Cause Analysis
description: Systematically analyze the causes of failures
category:
- Process
quality_tactics_url: https://qualitytactics.de/en/reliability/root-cause-analysis
problems:
- constant-firefighting
- high-defect-rate-in-production
- partial-bug-fixes
- regression-bugs
- slow-incident-resolution
- delayed-issue-resolution
- increased-error-rates
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Conduct structured root cause analysis after every significant production incident using techniques like the "5 Whys" or fishbone diagrams
- Distinguish between symptoms, contributing factors, and true root causes before implementing fixes
- Involve cross-functional participants including developers, operations, and business stakeholders
- Document findings in a shared knowledge base accessible to all team members
- Track root cause categories over time to identify systemic patterns (e.g., most incidents caused by legacy database issues)
- Create actionable follow-up items with owners and deadlines for each root cause identified
- Review the effectiveness of implemented fixes by checking whether similar incidents recur

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Breaks the cycle of recurring incidents by addressing causes rather than symptoms
- Builds organizational knowledge about legacy system failure modes
- Drives systemic improvements that reduce overall incident frequency
- Creates a feedback loop between production operations and development practices

**Costs and Risks:**
- Analysis takes time away from feature development and immediate firefighting
- Blame-oriented analysis discourages honest participation and hides real causes
- Root cause analysis can lead to analysis paralysis if not time-boxed
- Not all root causes are practical to fix in legacy systems, requiring risk acceptance decisions

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A logistics company experienced weekly outages in its legacy routing system. Each time, the team applied a quick fix and moved on. After implementing mandatory root cause analysis, the team discovered that all incidents traced back to a single database table that lacked proper indexing for the query patterns used by a feature added two years prior. The query worked fine at low data volumes but degraded as the table grew. A single index addition eliminated an entire class of incidents that had consumed hundreds of engineering hours in firefighting over the previous year.
