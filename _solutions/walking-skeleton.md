---
title: Walking Skeleton
description: Develop a minimal, running system with the core architectural ideas
category:
- Architecture
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/walking-skeleton
problems:
- implementation-starts-without-design
- modernization-strategy-paralysis
- analysis-paralysis
- strangler-fig-pattern-failures
- immature-delivery-strategy
- complex-deployment-process
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify the thinnest possible end-to-end slice through the system that exercises all major architectural components
- Build a minimal but fully functional version that includes UI, business logic, data persistence, and deployment
- Use the walking skeleton to validate the deployment pipeline, infrastructure, and integration points early
- Prioritize proving architectural risks over delivering features in the initial skeleton
- Iterate on the skeleton by adding flesh: incrementally implement real features on top of the proven architecture
- When modernizing a legacy system, use the walking skeleton to prove the target architecture before migrating features
- Keep the skeleton deployable at all times to maintain a working reference point

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Validates architectural assumptions and deployment pipelines before significant investment
- Provides a tangible, running system early that stakeholders can see and interact with
- Surfaces integration issues between components at the earliest possible stage
- Reduces risk of discovering fundamental architectural flaws late in the project

**Costs and Risks:**
- The skeleton may be too thin to reveal certain architectural challenges
- Stakeholders may misinterpret the minimal prototype as the final product quality
- Requires discipline to keep the skeleton minimal rather than prematurely adding features
- May delay visible feature delivery in the short term

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A government agency planned to modernize a legacy permit processing system by migrating from a mainframe to a cloud-native architecture. Previous modernization attempts had stalled after months of design without working code. This time, the team built a walking skeleton: a single permit type flowing through a React frontend, a Spring Boot API, a PostgreSQL database, and a Kubernetes deployment pipeline. The skeleton processed exactly one permit type with minimal business logic, but it proved that the architecture worked end-to-end and that the deployment pipeline could deliver changes reliably. With the architectural risks retired, the team confidently began migrating the remaining 30 permit types onto the proven foundation.
