---
title: Consumer Driven Contracts
description: Contracts that define the expectations of interface users
category:
- Testing
- Architecture
quality_tactics_url: https://qualitytactics.de/en/compatibility/consumer-driven-contracts
problems:
- breaking-changes
- integration-difficulties
- poor-interfaces-between-applications
- api-versioning-conflicts
- inadequate-integration-tests
- fear-of-breaking-changes
- microservice-communication-overhead
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Have each consumer define a contract specifying exactly which fields, endpoints, and behaviors it relies on
- Use a contract testing tool (e.g., Pact) to verify provider changes against all registered consumer contracts
- Run contract tests in the provider's CI pipeline so breaking changes are caught before merge
- Start by adding contracts for the most critical or fragile integration points in the legacy landscape
- Store contracts in a shared broker or repository accessible to both consumer and provider teams
- Use contract tests to replace brittle end-to-end integration tests where possible

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Providers know exactly which parts of their interface consumers depend on, enabling safe evolution
- Catches breaking changes at build time rather than in production
- Enables independent deployment of services without coordinated release windows

**Costs and Risks:**
- Requires consumer teams to write and maintain their contracts, adding cross-team coordination
- Contract testing tools have a learning curve and infrastructure requirements
- Contracts only test the interface shape, not full integration behavior
- Stale contracts can give false confidence if consumer teams do not update them

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A travel booking platform had 12 microservices with frequent integration failures because backend changes unknowingly broke frontend expectations. The team introduced Pact-based consumer-driven contracts for the five most critical service boundaries. Within three months, the contract tests caught 14 would-be breaking changes during code review, and integration-related production incidents dropped from a weekly occurrence to roughly one per quarter.
