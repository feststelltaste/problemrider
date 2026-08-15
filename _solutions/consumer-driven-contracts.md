---
title: Consumer Driven Contracts
description: Contracts that define the expectations of interface users
category:
- Testing
- Architecture
problems:
- breaking-changes
- integration-difficulties
- poor-interfaces-between-applications
- api-versioning-conflicts
- inadequate-integration-tests
- fear-of-breaking-changes
- microservice-communication-overhead
- communication-risk-outside-project
- poor-contract-design
- rapid-system-changes
layout: solution
related_solutions:
- slug: contract-testing
  similarity: 0.85
- slug: api-first-development
  similarity: 0.75
- slug: backward-compatible-apis
  similarity: 0.7
- slug: design-by-contract
  similarity: 0.7
- slug: integration-tests
  similarity: 0.7
- slug: abstraction
  similarity: 0.65
---

## Description

Consumer-driven contracts invert the usual direction of interface testing: instead of a provider unilaterally deciding what its interface looks like and hoping consumers keep up, each consumer specifies exactly which fields, endpoints, and behaviors it actually depends on, and that specification becomes an executable contract the provider must satisfy on every change. Tools such as Pact run these contracts in the provider's CI pipeline, so a change that would silently break a consumer fails the build before it merges rather than after it reaches production. This matters most in legacy landscapes that have grown into many services with implicit, undocumented dependencies between them, where nobody on the provider side can enumerate every consumer's actual usage of an interface from memory, and where breaking changes historically surfaced only as production incidents. Because contracts capture the interface shape consumers rely on rather than full end-to-end behavior, they are cheaper to run and maintain than broad integration test suites, and they let teams replace some of those brittle integration tests outright. The approach does require consumer teams to write and keep their contracts current, and it only provides safety for the interactions that are actually under contract, so the practice is most valuable when applied first to the most fragile or business-critical integration points.

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
