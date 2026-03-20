---
title: Architecture Decision Records (ADR)
description: Documenting important architectural decisions and their justifications
category:
- Architecture
- Communication
quality_tactics_url: https://qualitytactics.de/en/maintainability/architecture-decision-records-adr/
problems:
- accumulated-decision-debt
- decision-avoidance
- decision-paralysis
- delayed-decision-making
- implicit-knowledge
- tacit-knowledge
- information-decay
- poor-documentation
- knowledge-gaps
- incomplete-knowledge
- stagnant-architecture
- history-of-failed-changes
- analysis-paralysis
layout: solution
---

## How to Apply ◆

> In legacy systems, the absence of decision rationale forces every team member to reverse-engineer intent from code, so ADRs work best when introduced retroactively for the most impactful past choices and proactively for all future ones.

- Begin by writing retrospective ADRs for the decisions that cause the most confusion today: why the system uses a specific database, why a particular integration pattern was chosen, or why a component was split the way it was. Pair this work with the people who still remember the original reasoning.
- Store ADRs in the source code repository alongside the legacy code, not in a wiki or shared drive that ages separately. This keeps decisions discoverable through the same tooling developers already use.
- Mark decisions that are now known to be problematic as "Deprecated" rather than deleting them. In legacy contexts, understanding why a bad decision was made is often as valuable as the decision itself.
- When a modernization initiative revisits an old choice — migrating from a monolithic database, replacing a messaging protocol, splitting a module — write a new ADR that explicitly references the original and explains what changed. This creates a traceable record of how the architecture evolved.
- Use ADRs as a gatekeeping tool during modernization: no architectural change to the legacy system is approved unless an ADR is drafted, reviewed, and merged first. This prevents uninformed changes from adding new layers of hidden debt.
- Reference ADR numbers directly in code comments wherever a decision has a visible manifestation. Legacy codebases are full of surprising constructs; a comment pointing to the ADR explaining the constraint is far more durable than tribal knowledge.
- Adopt a lightweight format to reduce the barrier to adoption. In legacy teams with heavy workloads, a Y-statement or a short table-row decision log is more likely to be used consistently than a full five-section document.
- Integrate ADR review into the pull request process for any change touching core components of the legacy system, so senior engineers can flag when a proposed change conflicts with a previously documented constraint.

## Tradeoffs ⇄

> ADRs impose a writing discipline that legacy teams often resist, but the long-term cost of undocumented decisions in aging systems far exceeds the effort of recording them.

**Benefits:**

- Eliminates the repeated re-evaluation of settled choices, a pattern that plagues legacy teams as key personnel leave and institutional memory erodes.
- Gives new maintainers a structured entry point for understanding why the system looks the way it does, dramatically reducing onboarding time for complex legacy codebases.
- Prevents well-intentioned modernization changes from accidentally violating constraints that shaped the original design — for example, removing a workaround that compensates for a known external system limitation.
- Creates an audit trail that satisfies compliance requirements in regulated industries, where legacy systems often handle sensitive data under long-standing legal obligations.
- Improves the quality of architectural discussions during modernization by shifting debates from competing opinions to documented tradeoffs.

**Costs and Risks:**

- Retroactive ADR writing requires extracting rationale from people whose memories of decisions made years ago may be incomplete or inconsistent, resulting in ADRs that document best guesses rather than actual reasoning.
- Without consistent enforcement, the ADR directory becomes incomplete: recent, undocumented decisions create the same knowledge gaps the practice was meant to prevent.
- Over-broad ADRs that try to capture years of accumulated decisions in a short burst tend to be superficial, recording what was decided without the full context of why — which is the most valuable part.
- Teams under pressure to deliver legacy maintenance and modernization simultaneously may treat ADR writing as overhead and deprioritize it until the practice fades.
- ADRs that document constraints that have since become irrelevant can mislead new developers if statuses are not kept current, potentially causing them to preserve unnecessary complexity.

## How It Could Be

> The following scenarios illustrate how ADRs address the specific knowledge problems that accumulate in long-lived legacy systems.

A financial services company running a payment processing system originally built in the early 2000s scheduled a migration from synchronous REST calls to an asynchronous message queue. When the architect proposed the change, a senior engineer recalled that synchronous calls had been deliberately chosen because the downstream banking API at the time did not guarantee idempotency, and asynchronous retries risked duplicate transactions. No documentation of this constraint existed. The team spent two weeks investigating whether the constraint still applied before writing their first ADR capturing the original reasoning. The new ADR confirmed the banking API had since added idempotency keys, making the migration safe — and the two ADRs together told a complete story of the decision's evolution.

A government agency maintaining a legacy permit-management system struggled with high turnover among developers familiar with the system's unusual two-database architecture. The design used a normalized relational database for writes and a denormalized flat-file structure for reads, a pattern that confused every new hire who encountered it. After introducing ADRs, the team wrote a retrospective record documenting that the flat-file structure had been mandated by a regulatory reporting requirement that needed sub-second query times in 2009, before the current database infrastructure was available. Subsequent hires could read the ADR and understand within minutes why the architecture looked the way it did, rather than spending weeks discovering it through code reading.

A logistics company undertaking a strangler fig migration of its legacy order management system found that different teams were making inconsistent technology choices for the replacement services — one team chose Kafka for messaging, another chose RabbitMQ, a third was considering a database polling approach. The lack of any documented standard allowed each team to independently evaluate the same problem from scratch. After losing weeks to duplicate evaluation effort, the modernization lead introduced ADRs as a decision authority mechanism. The first cross-team ADR documented the choice of Kafka as the standard message broker, the context behind it, and the alternatives considered. Subsequent teams could adopt or formally challenge the decision rather than silently diverging from it.
