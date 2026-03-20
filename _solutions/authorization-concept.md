---
title: Authorization Concept
description: Defining access to critical data and functions
category:
- Security
- Management
quality_tactics_url: https://qualitytactics.de/en/security/authorization-concept
problems:
- authorization-flaws
- data-protection-risk
- regulatory-compliance-drift
- poorly-defined-responsibilities
- insufficient-audit-logging
- authentication-bypass-vulnerabilities
layout: solution
---

## How to Apply ◆

> Legacy systems often lack a documented authorization concept, resulting in ad-hoc permission assignments that accumulate over years. An authorization concept defines a clear model for who can access what data and functions, serving as the blueprint for implementation.

- Document all data classifications in the legacy system (public, internal, confidential, restricted) and map which user roles require access to each classification level. This creates the foundation for a permission model grounded in business requirements.
- Define functional permissions by mapping business processes to the system operations they require. For each role, list the specific create, read, update, and delete operations permitted on each resource type.
- Establish the principle of least privilege as a design rule: each role receives only the minimum permissions necessary to perform its business function. Document the rationale for each permission grant so it can be reviewed and challenged during audits.
- Create a role hierarchy that reflects the organizational structure and avoids permission inheritance that grants broader access than intended. Document which permissions are inherited and which are explicitly assigned.
- Define separation-of-duties rules that prevent a single user from performing conflicting operations (e.g., creating and approving a financial transaction). Implement these as hard constraints in the authorization system.
- Establish a formal process for requesting, granting, reviewing, and revoking access. Include mandatory approval workflows, time-limited access grants for temporary needs, and automatic revocation when users change roles.
- Schedule periodic access reviews (quarterly or semi-annually) where role owners verify that all assigned permissions are still necessary and appropriate.

## Tradeoffs ⇄

> A well-defined authorization concept provides clear, auditable access control that aligns with business needs, but it requires significant upfront design effort and ongoing governance.

**Benefits:**

- Provides a single authoritative document that defines all access rights, making security audits straightforward and efficient.
- Enables consistent implementation of access controls across the system by translating business requirements into technical permissions.
- Supports compliance with regulations that require demonstrable access control policies (GDPR, HIPAA, SOX, PCI DSS).
- Reduces the risk of privilege accumulation by establishing formal processes for granting and reviewing access.

**Costs and Risks:**

- Creating the initial authorization concept for a legacy system with years of ad-hoc permissions requires significant analysis effort to understand existing access patterns.
- Enforcing the concept may require removing permissions that users have grown accustomed to, causing resistance and potential workflow disruption.
- The authorization concept must be maintained as the system and organization evolve; an outdated concept provides false assurance.
- Overly complex role hierarchies can become difficult to understand and manage, creating new risks through misconfiguration.

## How It Could Be

> The following scenarios illustrate how an authorization concept brings order to access control in legacy systems.

A legacy insurance claims system has been in production for 12 years. Over that time, permissions have been granted on a per-request basis with no overarching model. An audit reveals that 40% of users have administrative access they do not need, including the ability to modify claim amounts and approve payments. The team creates a comprehensive authorization concept that defines five core roles (Claims Clerk, Claims Adjuster, Claims Manager, Auditor, System Administrator) with clearly documented permissions for each. Separation-of-duties rules prevent adjusters from approving their own claims. The team maps all 800 users to the appropriate roles, removing unnecessary administrative access from 320 accounts. The authorization concept is formalized as a living document with a quarterly review cycle, and the claims system's role structure is refactored to match the concept exactly.

A legacy government benefits system must comply with new data protection regulations that require demonstrable access controls for citizen data. The system has no documented authorization model — developers add permission checks ad-hoc when regulators raise specific concerns. The team develops an authorization concept that classifies all citizen data by sensitivity level, defines roles for each department that interacts with the system, and specifies which data fields each role may access. The concept includes data masking rules (e.g., Social Security numbers are displayed as XXX-XX-1234 to all roles except authorized case workers). Implementation reduces the number of users with access to unmasked sensitive data from 200 to 35, and the documented concept passes regulatory review on the first attempt.
