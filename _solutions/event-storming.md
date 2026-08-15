---
title: Event Storming
description: Discovering domain events, commands, and aggregates in collaborative
  workshops
category:
- Requirements
- Architecture
problems:
- legacy-business-logic-extraction-difficulty
- implicit-knowledge
- requirements-ambiguity
- stakeholder-developer-communication-gap
- poor-domain-model
- monolithic-architecture-constraints
layout: solution
related_solutions:
- slug: domain-modeling
  similarity: 0.75
- slug: business-process-modeling
  similarity: 0.7
- slug: architecture-workshops
  similarity: 0.7
- slug: domain-driven-design
  similarity: 0.7
- slug: story-mapping
  similarity: 0.65
- slug: bounded-contexts
  similarity: 0.65
---

## Description

Event storming is a collaborative workshop format, developed within the Domain-Driven Design community, in which developers, domain experts, and stakeholders jointly reconstruct a business process using sticky notes: first placing domain events in chronological order, then adding the commands that trigger them, the aggregates responsible for handling those commands, and the policies that connect one event to the next automatically. The technique is particularly effective against a specific legacy problem — that the actual behavior of an old system is often known only in fragments, scattered across the heads of a few long-tenured people, with no single artifact describing the full process end to end. Because the workshop format surfaces this knowledge collectively and visually, in a matter of hours rather than the weeks a written specification effort might take, it tends to reveal contradictions, undocumented side channels, and gaps in the team's shared understanding that no individual participant knew about beforehand. The clusters of events and aggregates that emerge from the session also double as natural candidate boundaries for decomposing a monolith, making event storming as valuable for planning a modernization's target architecture as it is for understanding the current one — though its output is only as durable as the effort put into formalizing it afterward, since sticky notes on a wall are not lasting documentation.

## How to Apply ◆

- Organize workshops with developers, domain experts, and stakeholders using sticky notes on a large wall or digital whiteboard.
- Start by identifying domain events (things that happen in the business) and arrange them chronologically.
- Add commands (what triggers events), aggregates (entities responsible for handling commands), and policies (automated reactions to events).
- Use the resulting event flow to map the legacy system's actual business processes, revealing hidden complexity and undocumented flows.
- Identify bounded context boundaries where different groups of events and aggregates form cohesive clusters.
- Use event storming output to guide decomposition of monolithic legacy systems into well-defined modules or services.

## Tradeoffs ⇄

**Benefits:**
- Rapidly surfaces implicit domain knowledge that exists only in people's heads.
- Creates shared understanding across business and technical participants in hours rather than weeks.
- Reveals gaps and contradictions in the current understanding of legacy system behavior.
- Produces natural boundaries for system decomposition and team organization.

**Costs:**
- Requires availability of key domain experts and developers for concentrated workshop time.
- Workshop output needs to be formalized and maintained; sticky notes alone are not lasting documentation.
- Facilitation skills are important; poorly facilitated sessions can be unproductive.
- Large legacy systems may require multiple sessions to cover adequately.

## How It Could Be

A legacy order fulfillment system needs to be decomposed for modernization, but no one has a complete picture of how all the pieces fit together. The team runs a two-day event storming workshop with warehouse managers, customer service representatives, and developers. They discover over sixty domain events and identify three distinct bounded contexts: order intake, warehouse operations, and shipping coordination. The workshop reveals that the legacy system handles returns through an undocumented side channel that bypasses the main order flow, a critical business process that was unknown to the development team. The event storming output becomes the blueprint for the decomposition effort, and the discovered bounded contexts guide both the technical architecture and the team structure for the modernization project.
