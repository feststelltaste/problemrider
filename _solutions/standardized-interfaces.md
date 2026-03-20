---
title: Standardized Interfaces
description: Adopt widely accepted interface styles so that any consumer can integrate without bespoke adapters
category:
- Architecture
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/compatibility/standardized-interfaces
problems:
- poor-interfaces-between-applications
- integration-difficulties
- rest-api-design-issues
- vendor-lock-in
- technology-lock-in
- legacy-api-versioning-nightmare
- tight-coupling-issues
layout: solution
---

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

## Examples

A logistics company has dozens of internal systems communicating through a mix of SOAP, FTP file drops, and custom TCP protocols. New consumer teams spend weeks building bespoke adapters for each integration. The architecture team introduces an API gateway that exposes RESTful OpenAPI-documented endpoints in front of the most critical legacy systems. Consumer teams now integrate using standard HTTP clients and auto-generated SDKs. Over time, the legacy backends are replaced with modern implementations behind the same standardized interfaces, and consumers experience no disruption during the transition.
