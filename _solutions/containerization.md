---
title: Containerization
description: Encapsulating applications and their dependencies in containers
category:
- Operations
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/containerization
problems:
- deployment-environment-inconsistencies
- configuration-drift
- dependency-version-conflicts
- complex-deployment-process
- poor-system-environment
- technology-stack-fragmentation
- deployment-risk
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Package each legacy application with its exact runtime, libraries, and configuration into a container image
- Use multi-stage builds to keep container images small while including all build-time dependencies
- Replace environment-specific installation scripts with declarative Dockerfiles
- Run the same container image across development, staging, and production to eliminate environment drift
- Introduce container orchestration (e.g., Kubernetes) gradually, starting with stateless services
- Use containers to run legacy applications side-by-side with modern services during migration

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates "works on my machine" problems by packaging the complete runtime environment
- Enables legacy applications to run on modern infrastructure without rewriting
- Simplifies dependency management by isolating each application's dependency tree
- Facilitates incremental modernization by allowing old and new services to coexist

**Costs and Risks:**
- Containerizing legacy applications with specific OS or hardware dependencies can be challenging
- Adds operational complexity through container orchestration, networking, and storage management
- Stateful legacy applications require careful handling of persistent storage in containers
- Teams need new skills in container tooling and orchestration platforms

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A manufacturing company ran a legacy inventory system on a specific version of Red Hat with pinned library versions. Server hardware refresh threatened to break the application. By containerizing the application with its exact dependency tree, the team decoupled it from the host OS, enabling deployment on modern infrastructure. The containerized application also became the foundation for a Strangler Fig migration, with new microservices deployed alongside the legacy container in the same Kubernetes cluster.
