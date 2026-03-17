---
title: Monitoring Gaps
description: Insufficient production monitoring and observability make it difficult
  to detect and diagnose issues in a timely manner, leading to longer outages and
  more severe consequences.
category:
- Code
- Process
related_problems:
- slug: quality-blind-spots
  similarity: 0.7
- slug: feature-gaps
  similarity: 0.6
- slug: poor-operational-concept
  similarity: 0.6
- slug: knowledge-gaps
  similarity: 0.55
- slug: system-integration-blindness
  similarity: 0.55
- slug: maintenance-bottlenecks
  similarity: 0.55
layout: problem
---

## Description
Monitoring gaps are blind spots in a system's observability. They are areas of the system that are not being monitored, or are not being monitored effectively. Monitoring gaps can make it difficult to detect and diagnose problems in production, which can lead to longer outages and more severe consequences. They are a common problem in complex, distributed systems, where it can be difficult to get a complete picture of the system's health.

## Indicators ⟡
- The first sign of a problem is often a customer complaint.
- It takes a long time to diagnose the root cause of a problem.
- The team is often surprised by the system's behavior.
- There is a lack of visibility into the system's key performance indicators (KPIs).

## Symptoms ▲

- [Negative User Feedback](negative-user-feedback.md)
<br/>  When monitoring gaps mean customers discover problems before the team does, user frustration and complaints increase.

## Causes ▼
- [Poor Operational Concept](poor-operational-concept.md)
<br/>  Systems designed without considering operational needs naturally lack the instrumentation and monitoring needed for production observability.
- [Knowledge Gaps](knowledge-gaps.md)
<br/>  Teams that lack experience with production operations may not know what metrics and alerts are important to monitor.
- [Time Pressure](time-pressure.md)
<br/>  Under pressure to deliver features, teams skip implementing proper monitoring and observability in favor of functional requirements.
- [Insufficient Audit Logging](insufficient-audit-logging.md)
<br/>  Insufficient logging directly creates blind spots in system monitoring and observability.
- [Quality Blind Spots](quality-blind-spots.md)
<br/>  Blind spots in testing often extend to monitoring, leaving production issues undetected as well.

## Detection Methods ○
- **Monitoring Coverage Analysis:** Analyze your monitoring tools to identify gaps in your coverage.
- **Incident Post-Mortems:** Review your incident post-mortems to identify cases where a lack of monitoring made it difficult to diagnose a problem.
- **Developer Interviews:** Ask developers about their experience with monitoring. Their feedback can be a valuable source of information.
- **Chaos Engineering:** Intentionally inject failures into your system to see how it behaves and to identify gaps in your monitoring.

## Examples
A company is running a microservices-based application. The application is complex, and it is difficult to get a complete picture of its health. The team does not have a good monitoring strategy, and they are not using the right tools. As a result, they are often surprised by production issues, and it takes them a long time to diagnose the root cause of problems. This has led to a number of long outages, which have had a significant impact on the business.
