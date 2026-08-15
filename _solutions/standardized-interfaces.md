---
title: Standardized Interfaces
description: Adopt widely accepted interface styles so that any consumer can integrate
  without bespoke adapters
category:
- Architecture
- Dependencies
problems:
- poor-interfaces-between-applications
- integration-difficulties
- rest-api-design-issues
- vendor-lock-in
- technology-lock-in
- legacy-api-versioning-nightmare
- tight-coupling-issues
- dependency-on-supplier
layout: solution
related_solutions:
- slug: standardized-protocols
  similarity: 0.75
- slug: consistent-user-interface
  similarity: 0.7
- slug: api-first-design
  similarity: 0.7
- slug: standardized-data-formats
  similarity: 0.7
- slug: data-formats
  similarity: 0.7
- slug: canonical-data-model
  similarity: 0.7
---

## Description

Standardized interfaces means replacing proprietary or ad hoc integration mechanisms — custom TCP protocols, SOAP endpoints with idiosyncratic conventions, file-drop integrations — with widely adopted interface styles such as REST, GraphQL, or gRPC, described using standard specification formats like OpenAPI or Protocol Buffers so that any consumer can integrate using common tools rather than a bespoke adapter built specifically for that one legacy system. Legacy landscapes tend to accumulate a different integration style for every system that was ever connected to them, and each new consumer team then has to invest weeks learning and coding against that system's particular quirks before any real integration work can begin. Introducing a facade or API gateway that exposes a standardized interface in front of the legacy implementation lets that cost be paid once, centrally, rather than repeatedly by every new consumer, and it decouples the consumer's integration effort from whatever the legacy backend happens to look like internally. This decoupling is what makes standardized interfaces valuable specifically during modernization: because consumers integrate against the stable, standardized contract rather than the legacy implementation directly, the backend behind that contract can be replaced incrementally without consumers needing to change anything. The corresponding cost is the upfront effort of building and governing that facade layer, and the risk that a generic standard interface cannot perfectly express every capability the legacy system originally offered, requiring deliberate compromises in the contract design.

## How to Apply ◆

- Replace proprietary or ad-hoc interfaces in legacy systems with industry-standard styles such as REST, GraphQL, or gRPC.
- Define interface contracts using standard specification formats (OpenAPI, Protocol Buffers, AsyncAPI) and publish them for consumers.
- Introduce an API gateway or facade in front of legacy systems to present standardized interfaces while the underlying implementation is migrated incrementally.
- Establish interface design guidelines that all teams follow, covering naming conventions, error formats, pagination, and authentication.
- Use contract testing to verify that both providers and consumers adhere to the agreed-upon interface specifications.
- Document all interfaces in a central API catalog so consumers can discover and integrate without ad-hoc communication.

## Tradeoffs ⇄

**Benefits:**
- Any consumer can integrate using well-known tools and libraries, reducing onboarding time.
- Decouples consumer and provider implementations, making independent evolution possible.
- Reduces the need for custom adapters, translators, and integration middleware.
- Makes it easier to replace legacy backend implementations without affecting consumers.

**Costs:**
- Wrapping legacy systems with standardized interfaces requires upfront development effort.
- Standard interfaces may not perfectly map to legacy system capabilities, requiring compromise or adaptation.
- Enforcing standards across autonomous teams requires governance and buy-in.
- Over-standardization can reduce flexibility for specialized use cases.

## How It Could Be

A logistics company has dozens of internal systems communicating through a mix of SOAP, FTP file drops, and custom TCP protocols. New consumer teams spend weeks building bespoke adapters for each integration. The architecture team introduces an API gateway that exposes RESTful OpenAPI-documented endpoints in front of the most critical legacy systems. Consumer teams now integrate using standard HTTP clients and auto-generated SDKs. Over time, the legacy backends are replaced with modern implementations behind the same standardized interfaces, and consumers experience no disruption during the transition.
