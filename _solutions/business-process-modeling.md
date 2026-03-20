---
title: Business Process Modeling
description: Elicit business requirements by modeling the underlying business processes
category:
- Requirements
- Business
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/business-process-modeling
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- legacy-business-logic-extraction-difficulty
- poor-domain-model
- stakeholder-developer-communication-gap
- implicit-knowledge
layout: solution
---

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
