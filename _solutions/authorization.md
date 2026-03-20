---
title: Authorization
description: Control access to resources based on permissions
category:
- Security
quality_tactics_url: https://qualitytactics.de/en/security/authorization
problems:
- authorization-flaws
- authentication-bypass-vulnerabilities
- data-protection-risk
- regulatory-compliance-drift
- password-security-weaknesses
- error-message-information-disclosure
layout: solution
---

## How to Apply ◆

> Legacy systems often implement authorization inconsistently — some endpoints check permissions while others do not, or authorization logic is scattered throughout the codebase with no central enforcement point. Systematic authorization ensures that every access to resources and operations is verified against a defined permission model.

- Map all protected resources and operations in the legacy system and define who should have access to each. Many legacy systems have accumulated ad-hoc access patterns where permissions were granted informally and never reviewed.
- Centralize authorization logic in a single enforcement layer rather than scattering permission checks across controllers, services, and database queries. A centralized approach ensures consistency and makes it possible to audit all access control decisions.
- Implement role-based access control (RBAC) as a baseline, mapping users to roles and roles to permissions. For more complex requirements, consider attribute-based access control (ABAC) that evaluates permissions based on user attributes, resource attributes, and environmental conditions.
- Add authorization checks at both the API/controller layer and the data access layer. API-level checks prevent unauthorized requests from being processed; data-level checks prevent unauthorized access through alternative paths (direct database queries, batch processes, reporting tools).
- Implement the deny-by-default principle: if no explicit permission grants access, the request is denied. Legacy systems often operate on an implicit-allow model where new features are accessible to all users unless someone remembers to add restrictions.
- Log all authorization decisions (both grants and denials) to support audit requirements and enable detection of unauthorized access attempts.
- Conduct periodic access reviews to remove permissions that are no longer needed, especially for users who have changed roles or left the organization.

## Tradeoffs ⇄

> Proper authorization ensures that users can only access resources and perform operations they are explicitly permitted to, but it requires comprehensive mapping of access requirements and consistent enforcement.

**Benefits:**

- Prevents unauthorized access to sensitive data and operations by enforcing explicit permission checks on every access path.
- Supports regulatory compliance requirements (GDPR, HIPAA, SOX) that mandate access control and the principle of least privilege.
- Provides audit capability by logging who accessed what and whether the access was authorized.
- Reduces the blast radius of compromised credentials — even if an attacker obtains valid credentials, they can only access resources permitted for that user's role.

**Costs and Risks:**

- Retrofitting authorization into a legacy system requires comprehensive identification of all access paths, which is time-consuming and error-prone for complex systems.
- Overly restrictive authorization can break existing workflows that relied on implicit access, causing disruption during rollout.
- Authorization logic must be maintained as the system evolves; new features that bypass the authorization layer reintroduce vulnerabilities.
- Complex permission models (hundreds of roles with fine-grained permissions) become difficult to manage and audit, creating their own security risks through misconfiguration.

## How It Could Be

> The following scenarios illustrate how authorization controls address access vulnerabilities in legacy systems.

A legacy document management system allows any authenticated user to access any document by modifying the document ID in the URL. The system checks that the user is logged in but does not verify that they have permission to view the requested document. The team implements resource-level authorization by associating each document with an access control list (ACL) and adding a permission check in the document retrieval service. They also add a centralized authorization middleware that intercepts all API requests, extracts the resource identifier, and verifies that the authenticated user's role grants the required permission. A security scan after deployment confirms that document enumeration attacks are no longer possible — unauthorized requests receive a 403 Forbidden response without revealing whether the document exists.

A legacy ERP system has accumulated 350 user roles over 15 years, many of which grant overlapping or excessive permissions. Several roles created for temporary projects still grant full administrative access. The team conducts a role consolidation exercise, mapping actual usage patterns from access logs to identify which permissions each user actually needs. They reduce the role count from 350 to 45 well-defined roles, implement deny-by-default authorization, and remove administrative privileges from 12 accounts that had them unnecessarily. Access reviews are scheduled quarterly to prevent permission accumulation from recurring.
