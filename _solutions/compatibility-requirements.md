---
title: Compatibility Requirements
description: Make implicit compatibility assumptions explicit and binding
category:
- Requirements
- Process
quality_tactics_url: https://qualitytactics.de/en/compatibility/compatibility-requirements
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- breaking-changes
- integration-difficulties
- fear-of-breaking-changes
- implicit-knowledge
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Document which systems, versions, protocols, and data formats your system must remain compatible with
- Include compatibility requirements in user stories and acceptance criteria, not just functional requirements
- Derive test cases directly from compatibility requirements so they are verifiable
- Review compatibility requirements during architecture reviews and before major changes
- Maintain a living document of compatibility commitments accessible to all teams
- Involve integration partners in defining and validating compatibility requirements

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Prevents compatibility issues caused by unstated assumptions
- Gives developers clear guidance on what they must preserve during changes
- Creates a contractual basis for compatibility testing and validation

**Costs and Risks:**
- Gathering and maintaining compatibility requirements takes effort and cross-team coordination
- Overly rigid requirements can constrain necessary architectural evolution
- Requirements may become outdated if not regularly reviewed
- Stakeholders may resist making implicit assumptions explicit because it creates accountability

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A transportation company integrated with 15 partner systems but had never documented which protocol versions and data formats each partner required. When a routine upgrade broke three partner integrations, the incident review revealed that no compatibility requirements existed. The team spent two weeks documenting requirements for all partner integrations, added them to the architecture decision records, and created automated compatibility tests. No unplanned partner integration failures occurred in the following 12 months.
