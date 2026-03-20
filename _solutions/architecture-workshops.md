---
title: Architecture Workshops
description: Conduct regular workshops to evolve the software architecture
category:
- Architecture
- Team
quality_tactics_url: https://qualitytactics.de/en/maintainability/architecture-workshops
problems:
- stagnant-architecture
- knowledge-silos
- implicit-knowledge
- team-silos
- limited-team-learning
- architectural-mismatch
- modernization-strategy-paralysis
- resistance-to-change
layout: solution
---

## How to Apply ◆

> In legacy environments, architecture workshops break down knowledge silos and build shared understanding of both the current system and the target architecture.

- Schedule regular workshops (monthly or quarterly) where developers from different teams examine, discuss, and propose improvements to the system's architecture together.
- Use workshops to reverse-engineer and document poorly understood parts of the legacy architecture, combining knowledge from developers who understand different parts of the system.
- Include hands-on activities such as collaborative diagramming, architecture katas, or guided codebase exploration rather than passive presentations.
- Focus each workshop on a specific architectural concern (e.g., reducing coupling between two modules, designing an API boundary, evaluating a technology migration path) to keep discussions productive.
- Invite participants from different teams and experience levels to ensure diverse perspectives and to spread architectural knowledge across the organization.
- Document workshop outcomes and decisions, and track follow-up actions to ensure that workshop insights translate into actual improvements.

## Tradeoffs ⇄

> Architecture workshops build shared understanding and drive architectural improvement but require time investment and skilled facilitation.

**Benefits:**

- Breaks down knowledge silos by bringing together developers who understand different parts of the legacy system.
- Builds team-wide architectural awareness, reducing the risk that individual changes inadvertently degrade the overall architecture.
- Creates a forum for discussing and resolving architectural tensions that individual teams cannot address alone.
- Generates momentum for modernization by helping the team envision and plan the target architecture collaboratively.

**Costs and Risks:**

- Workshops consume development time from multiple team members simultaneously, which may be difficult to justify under delivery pressure.
- Without skilled facilitation, workshops can devolve into unfocused debates or complaint sessions that produce no actionable outcomes.
- Workshop decisions may not be implemented if there is no follow-up mechanism to track and prioritize resulting work items.
- Participants without sufficient context may contribute noise rather than signal, making the workshop less productive for experienced architects.

## How It Could Be

> The following scenario illustrates how architecture workshops advance legacy modernization.

A healthcare software company held quarterly architecture workshops where developers from five teams spent a full day working on architectural challenges. In one workshop, the teams collaboratively mapped all data flows between their legacy monolith's 14 modules, discovering three circular dependencies that no single team had been aware of. The workshop produced a concrete plan to break these cycles through the introduction of event-based communication, which the teams implemented over the following quarter. In another workshop, the teams evaluated two competing approaches for migrating the authentication module and reached consensus on an approach that none of the individual teams had considered. The workshops became the primary venue for cross-team architectural alignment and were credited with reducing inter-team integration issues by 40% over two years.
