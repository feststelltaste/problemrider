---
title: Strangler Fig Pattern
description: Replacing legacy systems incrementally by routing traffic to new implementations
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/strangler-fig-pattern/
problems:
- monolithic-architecture-constraints
- legacy-business-logic-extraction-difficulty
- strangler-fig-pattern-failures
- stagnant-architecture
- system-stagnation
- technology-lock-in
- fear-of-breaking-changes
- fear-of-change
- architectural-mismatch
- inability-to-innovate
- technical-architecture-limitations
- high-maintenance-costs
- obsolete-technologies
layout: solution
---

## How to Apply ◆

> In legacy system modernization, the Strangler Fig Pattern replaces the dangerous all-or-nothing rewrite with a controlled, capability-by-capability migration that keeps the legacy system fully operational throughout.

- Place a routing layer — a reverse proxy, API gateway, or message router — in front of the legacy system before writing a single line of replacement code. This layer is the foundational investment that all subsequent migrations will rely on; build it to support gradual traffic shifting and fast rollback.
- Map the legacy system's capabilities and identify natural migration boundaries using dependency analysis or event storming. Look for clusters of functionality that have limited data coupling to the rest of the system — these are the safest starting points.
- Prioritize capabilities that change frequently for early migration. A business rule that requires monthly updates to a poorly structured legacy module is a prime candidate: the team gains immediate relief from the legacy codebase where it hurts most.
- Before building any replacement, write characterization tests against the legacy system's actual behavior — including its undocumented quirks and edge cases. These tests become the acceptance criteria the new implementation must satisfy and the safety net during cutover.
- Use shadow traffic or canary deployments for each capability cutover: initially send a small percentage of real requests to both old and new implementations, compare outputs, and shift traffic fully only after discrepancies are resolved.
- Establish a strict discipline of removing legacy dead code immediately after each capability migrates successfully. Without this, the legacy system persists indefinitely alongside the new one, doubling the maintenance burden rather than reducing it.
- Plan for data migration as a first-class concern. Identify early whether capabilities can share the legacy database temporarily, require data synchronization, or can own their data independently from the moment of cutover.
- Set a firm timeline for completing each migration phase and communicate it across the teams involved. Without organizational commitment to decommission, the pattern becomes a permanent dual-system architecture rather than a transitional one.

## Tradeoffs ⇄

> The Strangler Fig Pattern trades the speed and simplicity of a clean-room rewrite for resilience and continuous delivery of value throughout a migration that may span years.

**Benefits:**

- Eliminates the existential risk of big-bang rewrites, which fail more often than they succeed in legacy modernization contexts where hidden complexity is the norm.
- Delivers visible, measurable modernization progress to stakeholders with each migrated capability, maintaining organizational momentum over a multi-year migration.
- Preserves the ability to roll back any individual capability to the legacy system if the new implementation reveals unexpected problems, limiting the blast radius of each migration step.
- Allows the team to learn from early migrations — establishing routing patterns, data migration techniques, and testing approaches — before tackling the most complex and entangled parts of the legacy system.
- Reduces the pressure to fully understand the legacy system before starting: the team learns it capability by capability, building knowledge incrementally alongside the new implementation.

**Costs and Risks:**

- Running two systems simultaneously increases infrastructure costs, operational complexity, and the cognitive load on the team responsible for both during the transition period.
- The routing layer introduces a new architectural component that must be monitored, maintained, and kept highly available, since all traffic flows through it.
- Data synchronization between legacy and new data stores is technically difficult and a frequent source of consistency bugs, especially for capabilities with high write volumes or complex transaction semantics.
- The migration can stall after early quick wins: the first capabilities to migrate are typically the cleanest, while deeply entangled modules at the core of the legacy system may prove extremely difficult to extract cleanly.
- Without a strong decommission mandate, teams may lose urgency to complete the migration once immediate pain is relieved, leaving both systems running indefinitely.

## Examples

> The following scenarios illustrate how the Strangler Fig Pattern plays out in real legacy modernization situations, from simple routing changes to complex data migrations.

A regional insurer running a monolithic policy management system built in the 1990s needed to modernize its customer self-service portal without disrupting daily operations. The team placed an API gateway in front of the legacy system and began migrating the simplest capability first: policy document retrieval. The new implementation pulled documents from a modern document store, while the gateway continued routing all other requests — premium calculations, endorsements, claims — to the legacy backend. Over eighteen months the team migrated one capability at a time, and the policy document retrieval migration gave them the template they applied to every subsequent one. The legacy system finally handled only the most deeply coupled accounting functions before the last migration was completed.

A European logistics company operating a freight dispatch system that mixed route planning, driver scheduling, and invoice generation into a single monolith faced growing demand from mobile applications for a modern REST API. The legacy system exposed only a proprietary thick-client protocol. The team built a thin adapter layer that translated REST calls to the legacy protocol, allowing mobile clients to connect while the real migration proceeded. Over the following two years, the route planning engine — the highest-value and most frequently modified component — was extracted into an independent service. The adapter layer was reconfigured to route planning requests directly to the new service while all other requests continued through the adapter into the legacy system, giving the company modern planning capabilities without interrupting dispatch operations.

A public utility managing a decades-old customer billing system attempted a full rewrite twice in ten years, both times abandoning the effort after two years of investment without a production release. On the third attempt, leadership mandated the strangler fig approach. The team began with meter-reading ingestion, which had clearly defined inputs and no shared state with the rest of the billing logic. After a successful migration of that capability, they moved to usage calculation for commercial accounts only — a smaller subset of the full problem. Each migration built confidence and refined the team's data synchronization approach. Three years later, the legacy billing core was finally decommissioned, and for the first time in the organization's history, the transition produced no billing disruption for customers.
