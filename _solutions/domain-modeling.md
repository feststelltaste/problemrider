---
title: Domain Modeling
description: Mapping domain concepts and relationships in a domain model
category:
- Architecture
- Requirements
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/domain-modeling
problems:
- poor-domain-model
- complex-domain-model
- legacy-business-logic-extraction-difficulty
- architectural-mismatch
- requirements-ambiguity
- stakeholder-developer-communication-gap
layout: solution
---

## How to Apply ◆

- Collaborate with domain experts to identify the key business concepts, their attributes, and relationships in the legacy system's domain.
- Create visual domain models (UML class diagrams, CRC cards, or informal diagrams) that represent the business domain independently of the current implementation.
- Compare the domain model against the legacy system's actual data structures and code organization to identify misalignments.
- Use the domain model to guide refactoring: restructure legacy code so that classes and modules correspond to domain concepts.
- Iterate on the domain model as understanding deepens; treat it as a living artifact, not a one-time document.
- Use domain models as a communication tool during planning sessions to ensure developers and stakeholders share the same understanding.

## Tradeoffs ⇄

**Benefits:**
- Creates a shared understanding of the business domain that bridges the gap between technical and business stakeholders.
- Reveals where the legacy system's structure diverges from the business reality it serves.
- Guides refactoring and restructuring efforts toward a more domain-aligned codebase.
- Serves as a foundation for applying domain-driven design patterns.

**Costs:**
- Building an accurate domain model requires significant time with domain experts.
- Domain models can become outdated if not maintained as the business evolves.
- The gap between the domain model and the legacy implementation may be too large to bridge incrementally.
- Over-modeling can slow down development if the team spends too much time perfecting the model.

## Examples

A legacy supply chain management system uses technical abstractions ("Record," "Transaction," "Item") that do not map to how logistics managers think about their domain ("Purchase Order," "Shipment," "Stock Keeping Unit"). The development team conducts domain modeling workshops with logistics managers and creates a domain model using business terminology. Comparing this model with the legacy code reveals that a single "Transaction" table stores purchase orders, shipment records, and inventory adjustments, distinguished only by a type code. This insight guides a refactoring effort that separates these concepts into distinct domain objects, making the code comprehensible to new developers and enabling the logistics team to communicate requirements using terms that map directly to code structures.
