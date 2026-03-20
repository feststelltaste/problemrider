---
title: Federated Identity (OAuth/OIDC)
description: Delegating authentication to trusted external identity providers
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/security/federated-identity
problems:
- authentication-bypass-vulnerabilities
- password-security-weaknesses
- session-management-issues
- data-protection-risk
- vendor-lock-in
- difficult-developer-onboarding
- technology-lock-in
layout: solution
---

## How to Apply ◆

> Legacy systems frequently maintain their own user databases with custom authentication logic, creating security risks from password storage, inconsistent authentication policies, and the burden of maintaining identity management. Federated identity delegates authentication to specialized, trusted identity providers.

- Evaluate the legacy system's current authentication mechanism and identify a migration path to OAuth 2.0 or OpenID Connect (OIDC). Many legacy systems use form-based login with session cookies that can be wrapped with an authentication proxy that handles the OAuth/OIDC flow.
- Deploy an identity provider (IdP) — either an organizational SSO solution (Azure AD, Okta, Keycloak) or a self-hosted OIDC provider — and configure it as the authoritative source for user identities.
- Implement an authentication proxy or middleware layer in front of the legacy application that intercepts unauthenticated requests, redirects to the IdP, and processes the returned tokens. This approach enables federation without modifying the legacy application's authentication code.
- Map federated identity attributes (roles, groups, email) to the legacy system's internal user model. This mapping layer translates between the IdP's token claims and the legacy application's authorization structure.
- Implement token validation for API access: replace legacy API key or basic auth mechanisms with OAuth 2.0 bearer tokens that are validated against the IdP's token introspection endpoint or by verifying JWT signatures.
- Plan a user migration from the legacy user database to the federated identity provider. This can be done gradually — allow login via both mechanisms during a transition period, then disable legacy authentication.
- Implement single sign-out so that terminating a session at the IdP also terminates sessions in the legacy application, preventing orphaned sessions.

## Tradeoffs ⇄

> Federated identity centralizes authentication at a specialized provider, improving security and user experience, but it introduces a dependency on the identity provider and requires integration work.

**Benefits:**

- Eliminates the need to store and manage passwords in the legacy system, removing a major category of security risk.
- Enables single sign-on (SSO) across the legacy system and other applications, improving user experience and reducing password fatigue.
- Centralizes authentication policy (password complexity, MFA requirements, lockout policies) at the identity provider, ensuring consistent enforcement.
- Simplifies user provisioning and deprovisioning by managing identities in a single location rather than in each legacy application separately.

**Costs and Risks:**

- The identity provider becomes a critical dependency — if it is unavailable, users cannot authenticate with the legacy system.
- Integrating OAuth/OIDC with legacy applications that use custom authentication can require significant middleware development.
- Token handling introduces new security considerations (token storage, refresh token rotation, token revocation) that the team must implement correctly.
- Legacy applications with embedded user databases may have authorization logic tightly coupled to their user model, making identity migration complex.

## How It Could Be

> The following scenarios illustrate how federated identity modernizes legacy system authentication.

A company operates 12 legacy applications, each with its own user database and login page. Users maintain separate credentials for each application, leading to widespread password reuse, frequent password reset requests, and inconsistent authentication security across applications. The team deploys Keycloak as a centralized OIDC identity provider and configures authentication proxies in front of each legacy application. Users now authenticate once through Keycloak and gain access to all applications via SSO. The legacy applications' user databases are retained for authorization mapping but no longer store passwords. MFA is enforced centrally at Keycloak for all applications. Password reset requests drop by 85%, and the consistent MFA requirement eliminates the credential-stuffing attacks that previously targeted the weakest application.

A legacy SaaS platform manages 50,000 user accounts with a custom authentication system that stores passwords as salted SHA-1 hashes and provides no MFA capability. Enterprise customers demand SAML/OIDC federation so their employees can use corporate credentials. The team integrates the platform with an OIDC-compatible identity broker that supports both SAML federation for enterprise customers and social login for individual users. Enterprise customers connect their corporate identity providers, eliminating the need for separate platform credentials. Individual users can migrate to social login or continue using local accounts (which are upgraded to bcrypt hashing). The platform's attack surface shrinks significantly as it no longer manages passwords for federated users.
