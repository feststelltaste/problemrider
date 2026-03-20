---
title: API Security
description: Securing APIs through rate limiting, schema validation, gateways, and token-based authentication
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/security/api-security
problems:
- rate-limiting-issues
- authentication-bypass-vulnerabilities
- authorization-flaws
- high-api-latency
- rest-api-design-issues
- sql-injection-vulnerabilities
- legacy-api-versioning-nightmare
- data-protection-risk
- cross-site-scripting-vulnerabilities
layout: solution
---

## How to Apply ◆

> Legacy APIs often lack fundamental security controls, having been designed for internal use behind firewalls that no longer provide adequate protection. API security hardens these interfaces against modern threats through layered controls.

- Deploy an API gateway in front of legacy API endpoints to centralize authentication, rate limiting, and request validation. The gateway acts as a security layer that can be configured without modifying the legacy application code.
- Implement token-based authentication (OAuth 2.0 or API keys with HMAC signatures) to replace any legacy authentication mechanisms such as basic auth over unencrypted connections or IP-based access control.
- Add rate limiting at the API gateway level to prevent abuse, brute-force attacks, and accidental denial of service from misbehaving clients. Configure per-client and per-endpoint limits based on expected usage patterns.
- Implement request schema validation to reject malformed or unexpected input before it reaches the legacy backend. This prevents injection attacks and protects against unexpected payloads that legacy code may not handle safely.
- Add response filtering at the gateway to prevent over-exposure of data. Legacy APIs often return entire database records including fields that the client does not need and should not see (internal IDs, audit fields, sensitive data).
- Enable mutual TLS (mTLS) for service-to-service API communication to ensure both parties are authenticated and traffic is encrypted, replacing legacy unencrypted internal communication patterns.
- Implement API versioning and deprecation policies so that insecure legacy API versions can be phased out while clients migrate to secured versions.

## Tradeoffs ⇄

> API security provides defense-in-depth for exposed interfaces, but introduces latency overhead and operational complexity that must be managed.

**Benefits:**

- Centralizes security controls at the API gateway, enabling protection of legacy APIs without modifying their source code.
- Prevents abuse through rate limiting and throttling, protecting backend systems from being overwhelmed by malicious or accidental overuse.
- Reduces the attack surface by validating inputs and filtering outputs before they reach or leave the legacy application.
- Enables gradual security improvement by adding controls incrementally without requiring a complete rewrite of legacy API endpoints.

**Costs and Risks:**

- API gateway adds latency to every request, which may be noticeable for latency-sensitive legacy applications.
- Token-based authentication requires clients to be updated to support the new authentication flow, which can be disruptive for external consumers.
- Overly restrictive rate limits or schema validation rules can break legitimate usage patterns, especially for legacy clients with non-standard request formats.
- The API gateway itself becomes a critical infrastructure component that must be highly available and properly secured.

## How It Could Be

> The following scenarios illustrate how API security protects legacy systems from modern threats.

A legacy CRM system exposes a REST API that was originally designed for internal use by a single frontend application. Over the years, the API has been shared with partners, mobile apps, and third-party integrations, all using basic authentication over HTTPS. The system has no rate limiting, and a partner's misconfigured batch job sends 50,000 requests per minute, causing the legacy database to become overloaded. The team deploys an API gateway that enforces OAuth 2.0 token authentication, applies per-client rate limits of 100 requests per minute, and validates request payloads against an OpenAPI schema. The gateway also strips sensitive fields (internal customer IDs, audit timestamps) from API responses. Within a month, the unauthorized batch job is blocked, two injection attack attempts are caught by schema validation, and the legacy backend's load stabilizes.

A legacy payment processing API accepts transaction requests with minimal input validation, relying on the calling application to send well-formed data. A security audit reveals that the API is vulnerable to SQL injection through a poorly sanitized merchant ID parameter. Rather than modifying the legacy codebase, the team configures the API gateway to validate all incoming parameters against strict patterns (merchant IDs must match a UUID format, amounts must be positive numbers, currency codes must be from an allowed list). Additionally, they implement a web application firewall (WAF) rule at the gateway that blocks common SQL injection patterns. These gateway-level protections are deployed within a week, compared to the estimated three months it would take to refactor the legacy input handling code.
