---
title: Domain-Based Authorization Concept
description: Control access to sensitive data based on business authorizations
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/domain-based-authorization-concept
problems:
- authorization-flaws
- data-protection-risk
- regulatory-compliance-drift
- secret-management-problems
- poor-domain-model
layout: solution
---

## How to Apply ◆

- Define authorization rules in terms of business roles and data ownership rather than technical permissions on system resources.
- Map the legacy system's current access control model against actual business authorization requirements to identify gaps and over-permissions.
- Implement attribute-based access control (ABAC) or role-based access control (RBAC) aligned with business domain concepts.
- Centralize authorization logic rather than scattering permission checks throughout the legacy codebase.
- Audit existing access patterns to discover users with excessive permissions accumulated over years of ad-hoc grants.
- Test authorization rules against business scenarios to ensure sensitive data is protected according to regulatory requirements.

## Tradeoffs ⇄

**Benefits:**
- Authorization rules reflect actual business policies, making them easier for business stakeholders to validate.
- Reduces the risk of unauthorized data access by aligning permissions with business intent.
- Supports regulatory compliance by providing auditable, business-meaningful access controls.

**Costs:**
- Retrofitting domain-based authorization into a legacy system with ad-hoc access controls is complex.
- Requires deep understanding of both the business domain and the legacy system's current permission model.
- Over-restrictive authorization can impede legitimate workflows if business roles are too narrowly defined.
- Centralized authorization becomes a critical component that must be highly available.

## How It Could Be

A legacy hospital information system grants database-level permissions to users, resulting in nurses having access to billing data and administrative staff seeing clinical records. Over the years, permissions accumulated without review, and no one is sure who has access to what. The team introduces a domain-based authorization model where access is controlled by clinical role (physician, nurse, pharmacist) and patient relationship (treating team, consulting, no relationship). Authorization rules are expressed in business terms: "Nurses on the patient's care team can view vital signs and medication orders but not billing information." The legacy system's scattered permission checks are consolidated into an authorization service. A comprehensive audit reveals and revokes hundreds of excessive permissions, significantly improving the hospital's compliance posture.
