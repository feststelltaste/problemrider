---
title: Immutable Infrastructure
description: Not modifying infrastructure components, but replacing them with new versions
category:
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/immutable-infrastructure
problems:
- configuration-drift
- deployment-environment-inconsistencies
- configuration-chaos
- deployment-risk
- complex-deployment-process
- frequent-hotfixes-and-rollbacks
- poor-system-environment
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Package legacy applications into machine images, containers, or deployment artifacts that include all dependencies
- Eliminate manual configuration changes on running servers; all changes must flow through the build pipeline
- Use infrastructure-as-code tools to define and version server configurations alongside application code
- Implement blue-green or canary deployment strategies where new versions replace rather than update existing instances
- Store application state externally (databases, object stores) so compute instances can be freely replaced
- Automate the creation of new infrastructure from scratch so that any environment can be rebuilt identically
- Tag and archive every deployed artifact for auditability and rollback capability

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates configuration drift that causes "works on my machine" and environment-specific bugs
- Makes deployments reproducible and auditable
- Simplifies rollback by redeploying the previous known-good artifact
- Reduces the risk of accumulated undocumented changes in production environments

**Costs and Risks:**
- Legacy applications with embedded state or local file dependencies require refactoring
- Build times increase since entire images must be rebuilt for each change
- Requires investment in automation tooling and container or image management infrastructure
- Teams accustomed to SSH-and-fix workflows need cultural and process adaptation

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A government agency ran a legacy Java application on servers that had accumulated years of manual configuration patches. No two servers were configured identically, and deployments frequently failed on some machines. The team containerized the application, capturing all dependencies and configuration into a Docker image built by CI. Deployments became simple image replacements, configuration drift vanished, and the team could reproduce any environment instantly. When a deployment caused issues, rolling back meant redeploying the previous image tag rather than attempting to reverse manual changes.
