---
title: Compatibility Requirements
description: Make implicit compatibility assumptions explicit and binding
category:
- Requirements
- Process
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- breaking-changes
- integration-difficulties
- fear-of-breaking-changes
- implicit-knowledge
- legal-disputes
layout: solution
related_solutions:
- slug: documentation-of-compatibility-requirements
  similarity: 0.85
- slug: compatibility-standards
  similarity: 0.8
- slug: compatibility-as-error
  similarity: 0.8
- slug: compatibility-governance
  similarity: 0.8
- slug: compatibility-measurement
  similarity: 0.8
- slug: compatibility-testing
  similarity: 0.8
---

## Description

Compatibility requirements are the explicit, written specification of which external systems, protocol versions, and data formats a system must remain compatible with, converting what is otherwise an unstated assumption carried only in the heads of a few long-tenured engineers into a documented, verifiable commitment. Once written down, these requirements can be attached directly to user stories and acceptance criteria rather than living only as an unstated backdrop to functional requirements, and test cases can be derived from them directly, making compatibility something the team actively verifies rather than something they merely hope they preserved. This documentation gap is common in legacy systems that integrate with many partner systems over long periods, where the specific protocol version or data format each partner depends on was understood informally at the time of original integration but was never captured anywhere durable, leaving current maintainers without any reliable way to know what a routine-looking change might break. When that gap exists, a seemingly ordinary upgrade can silently violate an assumption no one remembered to check, breaking multiple partner integrations simultaneously and only revealing the missing requirement during the incident review that follows. Reviewing compatibility requirements as a standing part of architecture reviews, and involving integration partners directly in defining and validating them, keeps the documented requirements aligned with what partners actually need rather than what the team assumes they need. The obvious cost is the ongoing effort of gathering and maintaining this documentation across many partners and keeping it current as needs evolve, and there is an organizational tendency to resist making implicit assumptions explicit precisely because doing so creates clear accountability where none existed before.

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
