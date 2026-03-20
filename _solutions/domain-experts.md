---
title: Domain Experts
description: Directly involve domain experts in development
category:
- Team
- Requirements
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/domain-experts
problems:
- stakeholder-developer-communication-gap
- requirements-ambiguity
- implicit-knowledge
- knowledge-gaps
- legacy-business-logic-extraction-difficulty
- poor-domain-model
layout: solution
---

## How to Apply ◆

- Embed domain experts directly in development teams rather than having them available only through formal request channels.
- Schedule regular sessions where domain experts walk developers through business processes and rules encoded in the legacy system.
- Have domain experts participate in code reviews of business logic changes to validate correctness.
- Use domain experts to verify that extracted legacy business rules are complete and accurate before reimplementing them.
- Create opportunities for informal knowledge transfer: pair programming sessions, whiteboard discussions, and desk-side consultations.
- Document domain knowledge captured from experts in a structured format to reduce bus-factor risk.

## Tradeoffs ⇄

**Benefits:**
- Reduces misunderstandings between business intent and technical implementation.
- Accelerates understanding of legacy business logic that may not be documented.
- Catches business logic errors during development rather than after deployment.
- Builds developer empathy for user needs and business constraints.

**Costs:**
- Domain experts' time is valuable and often limited; their involvement needs careful scheduling.
- Experts may have difficulty expressing their knowledge in terms developers can act on.
- Over-reliance on a single domain expert creates a knowledge bottleneck.
- Domain experts may describe idealized processes rather than the actual implemented behavior.

## How It Could Be

A legacy tax calculation system contains hundreds of business rules accumulated over two decades, but the original developers have left the company. A tax specialist is embedded in the development team during a modernization project. She identifies numerous cases where the legacy code implements rules that were superseded by regulatory changes years ago, as well as several edge cases where the code diverges from correct tax law. Her involvement prevents the team from faithfully replicating bugs into the new system and ensures that the modernized system correctly implements current regulations. The domain expert also helps the team establish a shared vocabulary for tax concepts, eliminating misunderstandings that had previously led to weeks of rework per sprint.
