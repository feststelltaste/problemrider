---
title: Technology Lock-In
description: A situation where it is difficult or impossible to switch to a new technology
  because of the high cost or effort involved.
category:
- Architecture
- Code
related_problems:
- slug: vendor-lock-in
  similarity: 0.75
- slug: technology-isolation
  similarity: 0.65
- slug: technology-stack-fragmentation
  similarity: 0.6
- slug: vendor-dependency-entrapment
  similarity: 0.6
- slug: system-stagnation
  similarity: 0.6
- slug: obsolete-technologies
  similarity: 0.6
layout: problem
---

## Description
Technology lock-in is a situation where it is difficult or impossible to switch to a new technology because of the high cost or effort involved. This is a common problem in monolithic architectures, where the entire system is built on a single technology stack. Technology lock-in can make it difficult to innovate, and it can also lead to high costs if the technology becomes obsolete or the vendor goes out of business.

## Indicators ⟡
- The entire system is built on a single technology stack.
- It is difficult or impossible to use new technologies in the system.
- The development team is not able to keep up with the latest technology trends.
- The system is expensive to maintain because of the high cost of the technology.

## Symptoms ▲

- [Technology Isolation](technology-isolation.md)
<br/>  Being locked into a specific technology stack prevents adoption of modern alternatives, isolating the system from current ecosystems.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Proprietary or outdated locked-in technologies often have high licensing and support costs.
- [Reduced Innovation](reduced-innovation.md)
<br/>  Inability to adopt new technologies limits the team's ability to innovate and leverage modern capabilities.
- [System Stagnation](system-stagnation.md)
<br/>  The inability to evolve the technology stack causes the system to stagnate and fall behind competitors.
## Causes ▼

- [Monolithic Architecture Constraints](monolithic-architecture-constraints.md)
<br/>  A monolithic architecture built on a single technology stack makes it impossible to incrementally adopt new technologies.
- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  When code is tightly coupled to specific technology APIs and patterns, switching technologies requires rewriting large portions of the system.
- [Accumulated Decision Debt](accumulated-decision-debt.md)
<br/>  Early technology decisions that were never revisited become deeply embedded, making change increasingly costly.
## Detection Methods ○
- **Technology Stack Analysis:** Analyze the technology stack of the system to identify which technologies are being used.
- **Developer Surveys:** Ask developers if they feel like they are able to use new technologies to improve the system.
- **Cost Analysis:** Analyze the cost of the technology to identify which technologies are the most expensive.

## Examples
A company has a large, monolithic e-commerce application that is built on a proprietary technology stack. The company is not able to use new technologies, such as cloud computing and microservices, because the system is not designed for them. As a result, the company is not able to innovate as quickly as its competitors. The company is also paying a lot of money for the proprietary technology, and they are locked into a single vendor.
