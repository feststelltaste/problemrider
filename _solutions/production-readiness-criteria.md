---
title: Production Readiness Criteria
description: Define what a component must provide — observability, recovery, ownership, documentation — before it is allowed to carry production load.
category:
- Operations
- Process
- Architecture
problems:
- rapid-prototyping-becoming-production
- immature-delivery-strategy
- quality-compromises
- monitoring-gaps
- operational-overhead
- inadequate-test-infrastructure
- constant-firefighting
- lack-of-ownership-and-accountability
- high-defect-rate-in-production
- log-spam
- insufficient-testing
- unclear-documentation-ownership
- environment-variable-issues
- no-formal-change-control-process
- database-connection-leaks
- inadequate-configuration-management
- incorrect-max-connection-pool-size
- legacy-configuration-management-chaos
- logging-configuration-issues
- misconfigured-connection-pools
- release-anxiety
- resource-allocation-failures
- service-discovery-failures
layout: solution
---

## Description

Production readiness criteria are an explicit checklist a component must satisfy before it is permitted to serve real users: it can be observed, it can be recovered, someone owns it, its failure modes are known, and its operational procedures are written. The criteria exist because of a specific and extremely common failure path — something is built quickly to demonstrate an idea, it works, it gets used, and it is in production before anyone decides that it should be. Nothing about it was designed to be operated: there are no metrics, no alerts, no runbook, and often no owner. Legacy systems are substantially composed of components that arrived this way, and the operational burden they impose is paid daily by whoever is on call. The criteria convert an implicit drift into an explicit decision, which can then be made deliberately — including the decision to accept a gap and record it.

## How to Apply ◆

> The components that hurt most in a legacy landscape are usually not the ones that were built badly, but the ones that were never meant to be permanent and were never subsequently made operable.

- **Write the criteria as a short checklist** covering the areas that determine operability: observability, failure behavior, recovery, ownership, dependencies, and documentation. Eight to twelve items is enough. A longer list is negotiated down item by item at exactly the moment when there is pressure to ship.
- Require **observability before launch**: the component emits metrics for its key operations, its logs are structured and correlatable, and there is at least one alert that fires when it is not doing its job. A component that can only be observed by a user complaining is the most expensive kind to operate.
- Require **a named owning team**, not an individual. Ownerless components are the ones that decay, and ownership assigned to a person who later leaves is ownership that quietly disappears.
- Require that **failure behavior is known and stated**: what happens when each dependency is unavailable, what the timeout and retry behavior is, and whether failure is graceful or total. Answering these questions before launch usually changes the design.
- Require **a recovery path** — how it is restarted, how it is rolled back, and whether its state can be restored. In a legacy landscape it is common to find components with no tested recovery procedure at all, and the discovery is always made at the worst time.
- Require a **runbook covering the known failure modes**, written by whoever built it. Two pages written at launch are worth vastly more than the reconstruction attempted during an incident at three in the morning.
- **Apply the criteria to existing components as well**, retrospectively and in priority order. Running the checklist against the components that generate the most incidents typically explains most of the incident load in one afternoon.
- Allow **explicit, recorded exemptions** with an owner and a date. The criteria's purpose is to make the gap a decision rather than an accident; a rigid gate with no exemption path gets bypassed entirely and then applies to nothing.
- **Verify rather than assert.** A checklist ticked without evidence measures optimism. Ask to see the dashboard, trigger the alert, execute the rollback in a lower environment.

## Tradeoffs ⇄

> The criteria prevent the slow accumulation of unoperable components, at the cost of slowing every launch and requiring an authority willing to enforce them.

**Benefits:**

- Prototypes stop becoming production systems by default, which is the origin of a large share of the operational burden in long-lived landscapes.
- Operational load declines measurably, because components arrive with the observability and recovery that would otherwise be added after the third incident, if at all.
- Ownership is established at the point of creation, when it is obvious who owns it, rather than reconstructed years later when it is not.
- The checklist applied retrospectively is an efficient diagnostic for an existing landscape, identifying quickly where the incident load comes from.
- On-call becomes more sustainable, which has a direct effect on retention among the people who carry it.

**Costs and Risks:**

- Every launch takes longer, and the cost is concentrated on small components where the overhead is proportionally largest.
- Enforcement requires authority. Criteria that can be overruled by whoever is in a hurry provide documentation of what should have happened and nothing else.
- A checklist encourages compliance over judgment: a component can satisfy every item and still be poorly designed for operation.
- Applied uniformly, the criteria impose the same burden on an internal tool as on a customer-facing service, which is disproportionate and breeds resentment.
- Retrospective application to a large legacy estate reveals more gaps than can be funded, which can be demoralizing without a prioritized plan attached.

## How It Could Be

A team inherited a landscape of 40-odd services accumulated over eight years, of which they discovered 14 had no alerting whatsoever and 6 had no identifiable owner. Their on-call rotation was averaging 11 pages a week, and roughly half of those were incidents discovered by users rather than by monitoring. They wrote a ten-item readiness checklist and applied it retrospectively, worst-first by incident count. The exercise itself was revealing: the single worst offender was a currency conversion service written as a two-week prototype in 2019, still running, with no metrics, no runbook, and a hardcoded retry loop that silently swallowed failures. Bringing the top eight components up to the criteria took a quarter. Pages fell from 11 a week to 3, and the proportion detected by monitoring rather than by users went from about half to over ninety percent.

Applying the criteria going forward stopped a repeat. A team built a demonstration of an automated document classification feature that stakeholders liked immediately, and the pressure to put it in front of real users within two weeks was substantial. Under the previous regime it would have gone live as-is. The readiness review found it had no failure handling for the classification service being unavailable, no metrics, and no owner beyond the individual who had built it. The team took an additional nine days to add those, and recorded two exemptions with dates — no load testing and no automated rollback — that were closed the following month. The classification service was unavailable for four hours six weeks later, and the feature degraded to manual routing with an alert, which the demonstration version would have handled by returning errors to users with no notification to anyone.
