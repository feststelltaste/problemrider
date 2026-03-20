---
title: Event Storming
description: Discovering domain events, commands, and aggregates in collaborative workshops
category:
- Requirements
- Architecture
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/event-storming
problems:
- legacy-business-logic-extraction-difficulty
- implicit-knowledge
- requirements-ambiguity
- stakeholder-developer-communication-gap
- poor-domain-model
- monolithic-architecture-constraints
layout: solution
---

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
