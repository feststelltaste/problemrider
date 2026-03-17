---
title: High Maintenance Costs
description: A disproportionately large amount of the development budget and effort
  is consumed by maintaining the existing system rather than creating new value.
category:
- Business
- Code
related_problems:
- slug: maintenance-cost-increase
  similarity: 0.8
- slug: maintenance-overhead
  similarity: 0.8
- slug: increased-cost-of-development
  similarity: 0.75
- slug: modernization-roi-justification-failure
  similarity: 0.6
- slug: maintenance-paralysis
  similarity: 0.6
- slug: maintenance-bottlenecks
  similarity: 0.6
layout: problem
---

## Description
High maintenance costs are a common problem for legacy systems. As a system ages, it becomes more and more expensive to maintain. This is because the codebase becomes more complex, the technology becomes obsolete, and the original developers leave the company. Eventually, the cost of maintaining the system can become so high that it is no longer economically viable. At this point, the company is faced with a difficult choice: either invest in a costly modernization project or continue to pour money into a dying system.

## Indicators ⟡
- The development team spends more than 50% of its time on maintenance tasks.
- The company is constantly deferring new projects because it can't afford to both maintain the old system and build new ones.
- The cost of fixing a bug is often higher than the cost of the original feature.
- The business is hesitant to approve any changes to the system because of the high cost and risk.

## Symptoms ▲

- [Inability to Innovate](inability-to-innovate.md)
<br/>  When most of the budget is consumed by maintenance, teams have no capacity to explore new technologies or build new features.
- [Modernization ROI Justification Failure](modernization-roi-justification-failure.md)
<br/>  High ongoing maintenance costs make it difficult to justify additional investment in modernization since budgets are already strained.
- [Customer Dissatisfaction](customer-dissatisfaction.md)
<br/>  Lack of new features and slow response to change requests frustrates customers as competitors deliver improvements.
- [High Turnover](high-turnover.md)
<br/>  Developers become frustrated working primarily on maintenance of aging systems rather than building new things, leading them to leave.
- [Maintenance Paralysis](maintenance-paralysis.md)
<br/>  When maintenance costs dominate the budget, the system enters a state where meaningful improvements become impossible.

## Causes ▼
- [High Technical Debt](high-technical-debt.md)
<br/>  Accumulated design shortcuts and code quality issues make every change more expensive and time-consuming.
- [Brittle Codebase](brittle-codebase.md)
<br/>  Maintaining a brittle codebase requires disproportionate effort as small changes demand extensive testing and fixing.
- [Cargo Culting](cargo-culting.md)
<br/>  Adopted technologies and patterns that the team doesn't understand become expensive to maintain and troubleshoot.
- [Cascade Failures](cascade-failures.md)
<br/>  Diagnosing and fixing cascade failure patterns requires extensive investigation across multiple components, increasing costs.
- [CV Driven Development](cv-driven-development.md)
<br/>  Unnecessarily complex technology choices driven by resume building create systems that are expensive to maintain after the original developer leaves.
- [Dependency on Supplier](dependency-on-supplier.md)
<br/>  Vendor-controlled components often come with escalating licensing and support costs that the organization cannot negotiate away.
- [Difficult Code Reuse](difficult-code-reuse.md)
<br/>  Maintaining multiple copies of similar code multiplies the effort needed for bug fixes and updates.
- [Feature Bloat](feature-bloat.md)
<br/>  Maintaining a large number of features, many rarely used, consumes disproportionate development resources.
- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  Constant production bug fixing diverts development resources from new features, increasing overall maintenance burden.
- [Modernization ROI Justification Failure](modernization-roi-justification-failure.md)
<br/>  Without modernization investment, legacy systems accumulate technical debt that drives maintenance costs ever higher.
- [Modernization Strategy Paralysis](modernization-strategy-paralysis.md)
<br/>  Delayed modernization decisions allow technical debt to compound, steadily increasing the cost of maintaining deteriorating systems.
- [Obsolete Technologies](obsolete-technologies.md)
<br/>  Maintaining systems built on obsolete technologies requires specialized knowledge and custom workarounds, driving up costs.
- [Poor Encapsulation](poor-encapsulation.md)
<br/>  Poor encapsulation makes the system more expensive to maintain because any internal change can break external consumers.
- [Ripple Effect of Changes](ripple-effect-of-changes.md)
<br/>  The amplified effort required for every change drives up the cost of maintaining and evolving the system.
- [Scaling Inefficiencies](scaling-inefficiencies.md)
<br/>  When systems cannot be scaled independently, organizations must overprovision resources, leading to disproportionately high infrastructure and maintenance costs.
- [Strangler Fig Pattern Failures](strangler-fig-pattern-failures.md)
<br/>  Managing both legacy and new components simultaneously doubles operational overhead and maintenance effort.
- [Suboptimal Solutions](suboptimal-solutions.md)
<br/>  Suboptimal designs require ongoing workarounds, patches, and support that inflate maintenance costs.
- [Tangled Cross-Cutting Concerns](tangled-cross-cutting-concerns.md)
<br/>  Maintaining cross-cutting logic scattered throughout the codebase requires disproportionate effort for any change.
- [Technology Isolation](technology-isolation.md)
<br/>  Custom solutions must be built for problems that have standard solutions in modern ecosystems, increasing maintenance costs.
- [Technology Lock-In](technology-lock-in.md)
<br/>  Proprietary or outdated locked-in technologies often have high licensing and support costs.
- [Technology Stack Fragmentation](technology-stack-fragmentation.md)
<br/>  Maintaining multiple incompatible technology stacks with separate tools, processes, and expertise is significantly more expensive than a standardized environment.
- [Workaround Culture](workaround-culture.md)
<br/>  Maintaining multiple layers of workarounds requires significantly more effort than maintaining properly designed solutions.

## Detection Methods ○
- **Cost of Ownership Analysis:** Calculate the total cost of owning and maintaining the system over its lifetime. This will give you a clear picture of the financial impact of the system.
- **Maintenance vs. New Development Ratio:** Track the percentage of the development budget that is spent on maintenance versus new development. A high ratio is a clear sign of a problem.
- **Bug Fix Cost Analysis:** Analyze the cost of fixing bugs over time. A rising cost is a sign that the system is becoming more difficult to maintain.
- **Business Value Assessment:** Assess the business value that the system is providing. If the cost of maintaining the system is greater than the value it is providing, it is time to consider decommissioning it.

## Examples
A large financial institution is running its core banking system on a mainframe that is over 30 years old. The system is written in COBOL, and it is becoming increasingly difficult and expensive to find developers who are proficient in the language. The company is spending millions of dollars a year just to keep the system running. They are unable to invest in new, innovative products because all of their resources are tied up in maintaining the old system. The company is stuck in a difficult position: they know that they need to modernize their system, but they are afraid of the cost and risk of such a large project.
