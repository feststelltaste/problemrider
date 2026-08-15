---
title: Domain Experts
description: Directly involve domain experts in development
category:
- Team
- Requirements
problems:
- stakeholder-developer-communication-gap
- requirements-ambiguity
- implicit-knowledge
- knowledge-gaps
- legacy-business-logic-extraction-difficulty
- poor-domain-model
- inappropriate-skillset
layout: solution
related_solutions:
- slug: domain-modeling
  similarity: 0.75
- slug: domain-patterns
  similarity: 0.7
- slug: on-site-customer
  similarity: 0.7
- slug: subject-matter-reviews
  similarity: 0.7
- slug: code-reviews
  similarity: 0.7
- slug: domain-quiz
  similarity: 0.7
---

## Description

Involving domain experts directly means embedding people with deep business knowledge inside the development team's daily work — reviews, pairing sessions, walkthroughs — rather than treating them as a resource reachable only through a formal request channel when a question happens to come up. This is particularly consequential for legacy systems, where the people who originally encoded the business rules into the software have frequently left the organization, leaving behind logic that runs correctly but whose rationale, edge cases, and assumptions are no longer known to anyone actively maintaining the system. A domain expert working alongside developers can validate whether extracted or reimplemented business rules are actually complete and correct, and can catch the specific and costly failure mode where a legacy implementation faithfully encodes a rule that was superseded by a regulatory or business change years ago but never updated in the code. Their presence also closes the stakeholder-developer communication gap in real time, during design and implementation, rather than after a feature has already shipped with a misunderstanding baked in. Because expert time is scarce and a single expert's account can still reflect an idealized rather than actual process, their knowledge should be captured in structured, durable documentation as it is transferred, reducing the risk that critical understanding once again becomes concentrated in one person who might eventually leave.

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
