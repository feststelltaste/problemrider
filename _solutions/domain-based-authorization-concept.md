---
title: Domain-Based Authorization Concept
description: Control access to sensitive data based on business authorizations
category:
- Security
- Architecture
problems:
- authorization-flaws
- data-protection-risk
- regulatory-compliance-drift
- secret-management-problems
- poor-domain-model
- authorization-role-explosion
layout: solution
related_solutions:
- slug: authorization-concept
  similarity: 0.85
- slug: authorization
  similarity: 0.8
- slug: role-based-access-control
  similarity: 0.75
- slug: least-privilege
  similarity: 0.65
- slug: role-model-rationalization
  similarity: 0.6
- slug: domain-modeling
  similarity: 0.6
---

## Description

A domain-based authorization concept defines access control rules in terms of business roles, responsibilities, and data ownership — who is treating this patient, who owns this order — rather than in terms of low-level technical permissions attached directly to database tables, columns, or system resources. This reframing matters because legacy systems frequently grew their permission models opportunistically over many years, granting access at whatever technical layer was convenient at the time, which produces exactly the kind of accumulated, un-auditable over-permissioning that no one can fully account for after enough time has passed. Expressing authorization in business terms instead means each rule can be validated directly against an actual business policy by someone who understands that policy, rather than requiring a technical translation step that introduces both error and ambiguity. Centralizing this logic — rather than leaving permission checks scattered throughout the legacy codebase wherever a developer once decided a check was needed — also makes the resulting rules auditable as a single artifact, which is essential for regulatory compliance in domains like healthcare or finance. Retrofitting this model onto a legacy system requires first mapping the system's existing, often undocumented access patterns against what the business actually intends, a process that reliably surfaces years of excessive permissions granted through ad-hoc requests that were never revisited or revoked.

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
