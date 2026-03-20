---
title: Anti Corruption Layer
description: Protect existing systems from negative influences of external systems
category:
- Architecture
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/compatibility/anti-corruption-layer
problems:
- architectural-mismatch
- poor-interfaces-between-applications
- integration-difficulties
- vendor-dependency
- vendor-dependency-entrapment
- vendor-lock-in
- technology-lock-in
- legacy-api-versioning-nightmare
- api-versioning-conflicts
- shared-dependencies
- cross-system-data-synchronization-problems
- breaking-changes
layout: solution
---

## How to Apply ◆

> In legacy integration work, the Anti Corruption Layer is the primary defense against letting an old system's data model, naming conventions, and error codes bleed into new code.

- Identify all points where your new or modernized code must talk to a legacy system — mainframe interfaces, COBOL copybook formats, aging SOAP endpoints, proprietary flat-file feeds — and make each one an explicit ACL boundary rather than a direct call.
- Build a dedicated module or package for each ACL; never let domain code import external API clients directly, because even one direct import starts the corruption.
- Implement a translator for every external concept: map the legacy system's opaque status codes, abbreviated field names, and fixed-width records to meaningful domain types before anything else in the codebase sees them.
- Write integration tests using recorded real responses from the legacy system; when the legacy interface changes (and it will), those tests will fail before production traffic is affected.
- Use the ACL to centralize all legacy-specific error handling — retries, timeouts, CICS ABEND codes, DB2 SQL codes — so that new services never need to know what an `SQLCODE -911` means.
- When the legacy system publishes a new interface version alongside the old one, add a second adapter and translator inside the ACL and use a configuration switch to control which version is active; this allows parallel operation without changing a single line of domain code.
- Add a circuit breaker inside the ACL for legacy back ends that are slow or unreliable, returning cached or degraded responses rather than propagating latency into modern services.
- Monitor translation failures as a separate operational metric; a rising failure rate almost always signals an undocumented change in the legacy system's output.

## Tradeoffs ⇄

> The ACL adds a layer of code to write and maintain, but in legacy contexts that cost is almost always outweighed by the protection it provides against the system's most stubborn form of decay — accumulated model corruption.

**Benefits:**

- Keeps the new codebase clean and internally consistent even when it must communicate with multiple legacy systems that each use different field names, data formats, and status codes.
- Concentrates all legacy-specific knowledge in one place, making it far easier to understand, test, and eventually remove when the legacy system is retired.
- Enables legacy back ends to be replaced or upgraded without touching domain code — only the adapter and translator in the ACL need to change.
- Provides a natural seam for strangler-fig migration: new functionality can be added to the modern system while the ACL continues to bridge the gap to legacy components.
- Protects the team from undocumented legacy behavior changes by surfacing them as ACL validation or translation failures rather than silent data corruption downstream.

**Costs and Risks:**

- Each legacy integration requires designing, building, and maintaining its own ACL, which adds development effort that teams under modernization pressure may underestimate.
- If the legacy system's model changes frequently (common in systems still under active maintenance), the translators inside the ACL require constant updates and can become a bottleneck.
- Developers unfamiliar with the pattern — especially those who grew up with the legacy system — may bypass the ACL for convenience, recreating the model corruption the layer was designed to prevent.
- A poorly designed ACL that accumulates state or grows too large can itself become a legacy problem, inheriting the complexity it was supposed to shield against.
- Latency is added for every cross-boundary call; in high-throughput integration scenarios this overhead must be measured and managed.

## Examples

> The following scenarios illustrate how the ACL pattern resolves real integration pressures in legacy modernization programs.

A retail bank was replacing its customer-facing loan origination portal while keeping its mainframe-based core banking system unchanged. The mainframe spoke in EBCDIC-encoded fixed-width records with field names like `CUST-NO`, `LN-AMT-APPRVD`, and `DT-ORIG`, and used two-character reason codes to signal approval outcomes. Rather than mapping these structures directly into the new portal's domain model, the team built a loan gateway ACL that translated mainframe records into proper `LoanApplication` and `CreditDecision` domain objects. When the mainframe team later reorganized the record layout to accommodate a new product type, only the ACL's translator had to change — none of the portal's business logic was affected.

An insurance company integrating a new claims management system with three legacy policy administration systems discovered that each system used a different identifier for the same policyholder. One used a nine-digit account number, another used a Social Security Number, and the third used an internal sequential key. The team created a separate ACL for each system, each of which resolved its local identifier to the canonical `PolicyholderId` used throughout the new system. The ACLs also normalized the wildly different claim-status vocabularies — "PEND", "OPEN", "AO", "CLD" from three different systems — into a single `ClaimStatus` enumeration. When investigators later needed to trace a claim across all three systems, the ACLs provided a controlled, documented translation path rather than a maze of ad-hoc string comparisons.

A logistics company modernizing its parcel tracking platform consumed event feeds from four different carrier APIs. Each carrier represented shipment events differently: one returned ISO timestamps, another used Unix epoch milliseconds, a third sent dates in `MM/DD/YYYY` local time. One carrier described the same physical state — parcel held at customs — using three different event codes depending on which country the parcel was in. Rather than scattering carrier-specific parsing throughout the tracking service, the team built a carrier ACL per integration that normalized all events into a canonical `TrackingEvent` domain object before they entered the system. Onboarding a fifth carrier later required only adding a new ACL adapter; the rest of the platform remained untouched.
