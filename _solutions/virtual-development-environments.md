---
title: Virtual Development Environments
description: Providing development environments in virtual machines or containers
category:
- Operations
- Process
quality_tactics_url: https://qualitytactics.de/en/portability/virtual-development-environments
problems:
- inefficient-development-environment
- deployment-environment-inconsistencies
- difficult-developer-onboarding
- inadequate-onboarding
- inconsistent-onboarding-experience
- poor-system-environment
- configuration-drift
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Define the development environment as code using Docker Compose, Vagrant, or devcontainers
- Include all required services, databases, and dependencies in the virtual environment definition
- Store the environment definition in the same repository as the application code for versioning
- Provide scripts or Makefile targets for common operations (start, stop, reset, seed data)
- Use volume mounts for source code so developers can use their preferred IDE while the application runs in the container
- Document how to set up and use the virtual environment in the project README
- Keep the virtual environment aligned with production configurations to minimize environment parity issues

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- New developers can have a working environment in minutes rather than days
- Eliminates "works on my machine" problems by standardizing the development environment
- Ensures development environments match production more closely, catching issues earlier
- Enables developers to work on multiple projects with conflicting dependencies simultaneously

**Costs and Risks:**
- Containers and virtual machines consume significant local resources (CPU, memory, disk)
- Complex legacy systems with many services may require powerful developer machines
- Debugging inside containers can be more cumbersome than debugging locally
- Environment definitions require ongoing maintenance as dependencies and services evolve
- Performance of volume-mounted source code can be slow on some platforms

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A fintech company had a legacy monolith that required Oracle Database, Redis, RabbitMQ, and several microservices to be running locally for development. New developer setup took three to five days and was documented in a 40-page wiki that was perpetually outdated. The team created a Docker Compose environment with all dependencies preconfigured and a seed script that loaded test data. Onboarding dropped to under two hours. When a database version upgrade was needed, the team updated the Docker image tag, and every developer received the new version on their next `docker compose pull`. This also eliminated four persistent "works on my machine" bugs that had plagued the team for months.
