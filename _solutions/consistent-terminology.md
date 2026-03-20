---
title: Consistent Terminology
description: Use uniform terms throughout the software
category:
- Communication
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/consistent-terminology/
problems:
- user-confusion
- poor-user-experience-ux-design
- inconsistent-behavior
- poor-naming-conventions
- user-frustration
- difficult-developer-onboarding
- knowledge-gaps
- poor-documentation
layout: solution
---

## How to Apply ◆

> Legacy systems accumulate inconsistent terminology as different teams and developers add features over the years, using different words for the same concept. Standardizing terms reduces confusion for both users and developers.

- Create a glossary of domain terms that maps every concept to a single canonical name. Audit the legacy system's UI labels, error messages, documentation, and code comments to identify where the same concept is referred to by different names.
- Establish naming conventions for new development that reference the glossary, and enforce them through code review. Every new UI label, API field name, and documentation reference should use the canonical term.
- Prioritize terminology fixes in the areas of the application with the highest user traffic and the most support tickets related to confusion.
- Update database column names and API field names incrementally during routine maintenance. Use backward-compatible aliases during the transition period to avoid breaking integrations.
- Include the glossary in onboarding materials so new team members and new users learn the correct terminology from the start rather than inheriting the inconsistencies.
- Coordinate terminology changes with user-facing documentation and training materials to avoid creating even more confusion during the transition.

## Tradeoffs ⇄

> Consistent terminology improves comprehension across the entire system but requires coordinated effort to change entrenched naming.

**Benefits:**

- Reduces user confusion caused by seeing different labels for the same concept in different parts of the application.
- Accelerates developer onboarding because new team members can learn one set of terms rather than mentally mapping between multiple naming schemes.
- Improves searchability within the codebase and documentation when concepts have a single canonical name.
- Reduces support requests caused by users misunderstanding labels or entering data into the wrong fields.

**Costs and Risks:**

- Renaming established terms in a legacy system can temporarily confuse long-time users who have memorized the old labels, requiring change management communication.
- Changing database column names or API fields requires migration effort and coordination with all consumers of those interfaces.
- Achieving team-wide agreement on canonical terms can be time-consuming when different groups have strong preferences for their established vocabulary.
- Partial terminology updates create a transition period where both old and new terms coexist, potentially increasing confusion before it decreases.

## Examples

> Terminology inconsistency is one of the most common and most overlooked usability problems in legacy systems.

A legacy banking system uses "account number," "account ID," "client reference," and "portfolio code" interchangeably across different modules to refer to the same customer account identifier. New tellers regularly enter the wrong identifier in the wrong field because they cannot tell which label refers to which concept. The team creates a domain glossary with input from business analysts, agreeing that "Account Number" is the canonical term. They systematically update UI labels module by module during scheduled maintenance windows, starting with the most heavily used customer-facing screens. After updating the three most-used modules, teller error rates related to incorrect account identification drop by half, and the training manual shrinks by several pages because it no longer needs to explain the different terms.
