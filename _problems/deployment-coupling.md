---
title: Deployment Coupling
description: A situation where multiple components or services must be deployed together,
  even if only one of them has changed.
category:
- Architecture
- Operations
related_problems:
- slug: tight-coupling-issues
  similarity: 0.7
- slug: ripple-effect-of-changes
  similarity: 0.65
- slug: shared-dependencies
  similarity: 0.65
- slug: deployment-risk
  similarity: 0.6
- slug: deployment-environment-inconsistencies
  similarity: 0.6
- slug: missing-rollback-strategy
  similarity: 0.6
solutions:
- ci-cd-pipeline
- event-driven-architecture
- event-driven-integration
- microservices
- microservices-architecture
- modulith
- rolling-updates
- trunk-based-development
layout: problem
---

## Description
Deployment coupling is a situation where multiple components or services must be deployed together, even if only one of them has changed. This is a common problem in monolithic architectures, where all the components are tightly coupled and deployed as a single unit. Deployment coupling can lead to long release cycles, large and risky releases, and a great deal of anxiety for the development team.

## Indicators ⟡
- A small change to one component requires the entire system to be redeployed.
- It is not possible to deploy different components of the system independently.
- The deployment process is complex and error-prone.
- The development team is afraid to make changes to the system because they are afraid of breaking something.

## Symptoms ▲

- [Large, Risky Releases](large-risky-releases.md)
<br/>  When components must deploy together, releases accumulate many changes across multiple components, increasing size and risk.
- [Long Release Cycles](long-release-cycles.md)
<br/>  Coordinating deployments across coupled components extends the time between releases.
- [Deployment Risk](deployment-risk.md)
<br/>  Deploying multiple components simultaneously increases the chance of failure and makes rollbacks more complex.
- [Delayed Value Delivery](delayed-value-delivery.md)
<br/>  Individual features are held back until all coupled components are ready for deployment.
- [Fear of Change](fear-of-change.md)
<br/>  The complexity and risk of coupled deployments makes teams reluctant to make changes.
- [Release Anxiety](release-anxiety.md)
<br/>  Teams experience anxiety around deployments because coupled releases have more moving parts that can fail.
## Causes ▼

- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  Components that are tightly coupled at the code level necessarily require coordinated deployment.
- [Shared Database](shared-database.md)
<br/>  Components sharing a database must be deployed together when schema changes affect multiple services.
- [Monolithic Architecture Constraints](monolithic-architecture-constraints.md)
<br/>  Monolithic architectures inherently bundle all components into a single deployable unit.
- [Shared Dependencies](shared-dependencies.md)
<br/>  Shared libraries or services create deployment coupling when updates to the shared component require coordinated releases.
## Detection Methods ○
- **Deployment Process Mapping:** Map out the steps in the deployment process to identify bottlenecks and areas for improvement.
- **Component Dependency Analysis:** Analyze the dependencies between components to identify which components can be deployed independently.
- **Developer Surveys:** Ask developers if they feel like they are able to deploy their changes quickly and safely.

## Examples
A company has a large, monolithic e-commerce application. The application is composed of a number of different components, including a product catalog, a shopping cart, and a payment gateway. The components are all tightly coupled and deployed as a single unit. When the development team wants to make a change to the product catalog, they must redeploy the entire application. This is a time-consuming and risky process, and it often leads to problems. As a result, the company is only able to release new software once a month.
