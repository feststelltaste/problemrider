---
title: Content Negotiation
description: Letting clients and servers agree on format, language, and encoding via HTTP
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/compatibility/content-negotiation
problems:
- poor-interfaces-between-applications
- rest-api-design-issues
- integration-difficulties
- breaking-changes
- legacy-api-versioning-nightmare
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Implement server-side content negotiation using standard HTTP Accept, Accept-Language, and Content-Type headers
- Support multiple response formats (JSON, XML, CSV) through the same endpoint rather than creating format-specific endpoints
- Use content negotiation for API versioning via custom media types (e.g., application/vnd.company.v2+json)
- Add fallback behavior that returns a sensible default format when the client does not specify preferences
- Document supported media types and negotiation behavior in your API documentation
- Test content negotiation paths as part of your integration test suite

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables a single endpoint to serve multiple client needs without URL proliferation
- Supports gradual format migration by adding new formats alongside existing ones
- Follows HTTP standards, making the API more predictable for experienced consumers

**Costs and Risks:**
- Adds complexity to request handling and serialization logic
- Debugging can be harder when the same endpoint returns different formats
- Not all HTTP clients handle content negotiation correctly, especially older ones
- Caching behavior becomes more complex with Vary headers

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy enterprise application served data exclusively in XML. When mobile clients needed JSON responses, the team initially created duplicate endpoints. After implementing content negotiation, both XML and JSON consumers hit the same endpoints, with the server selecting the format based on the Accept header. This eliminated 30% of endpoint duplication and allowed the team to later add Protocol Buffers support for high-throughput internal consumers without any new routes.
