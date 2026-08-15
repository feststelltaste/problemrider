---
title: Business Process Modeling
description: Elicit business requirements by modeling the underlying business processes
category:
- Requirements
- Business
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- legacy-business-logic-extraction-difficulty
- poor-domain-model
- stakeholder-developer-communication-gap
- implicit-knowledge
- process-software-misfit
layout: solution
related_solutions:
- slug: data-modeling
  similarity: 0.75
- slug: requirements-analysis
  similarity: 0.75
- slug: business-process-automation
  similarity: 0.7
- slug: domain-modeling
  similarity: 0.7
- slug: user-stories
  similarity: 0.7
- slug: evolutionary-requirements-development
  similarity: 0.7
---

## Description

Business process modeling captures how a business process actually functions — through stakeholder interviews and direct observation of real workflows rather than reliance on existing documentation — and represents it visually, typically in BPMN, so both business and technical stakeholders share the same understanding of the process independent of any particular system's implementation. The mechanism's value lies specifically in the gap it is built to expose: documented procedures and actual practice diverge, often significantly, and only by observing what people actually do can that divergence be found and reconciled. Legacy systems are a common source of exactly this divergence, because users adapt to a system's limitations over years by developing informal workarounds that never make it into any specification, yet represent genuine business needs the system fails to meet directly. Mapping legacy system functionality against the resulting process model reveals which parts of the system support which processes and surfaces the workarounds as real requirements rather than as noise to be ignored during a replacement effort. Modernization projects that skip this step risk the most common failure mode of legacy replacement: building a new system that faithfully reproduces old software behavior instead of the actual business process the old software was only ever an imperfect vehicle for.

## How to Apply ◆

- Interview business stakeholders and observe actual workflows to capture how business processes really work, not just how documentation says they should work.
- Use BPMN or similar notation to create visual process models that both business and technical teams can understand.
- Map legacy system functionality to the business process model to identify which parts of the system support which processes.
- Identify discrepancies between documented processes and actual system behavior, which are common in legacy environments.
- Use process models to discover automation opportunities and redundant manual steps.
- Maintain process models as living documents that are updated when processes or requirements change.

## Tradeoffs ⇄

**Benefits:**
- Creates a shared understanding of business processes between business and technical stakeholders.
- Reveals hidden business logic embedded in legacy systems that may not be documented anywhere.
- Provides a foundation for requirements gathering during modernization efforts.
- Identifies inefficiencies and redundancies in current processes.

**Costs:**
- Modeling existing processes accurately requires significant time investment and stakeholder access.
- Process models can become outdated quickly if not actively maintained.
- Stakeholders may describe idealized rather than actual processes, requiring observation to validate.
- Over-detailed models can become as hard to understand as the code they describe.

## How It Could Be

A government agency plans to replace a legacy case management system but discovers that no one fully understands the current business processes. The team conducts workshops with case workers, creating BPMN diagrams of how cases actually flow through the system. They discover that the real process diverges significantly from the official procedures manual: case workers have developed numerous workarounds to compensate for system limitations. These workarounds represent genuine business needs that must be addressed in the replacement system. The process models become the authoritative requirements source for the modernization project, preventing the common mistake of building a new system that replicates old software behavior rather than actual business needs.
