---
title: Customizing
description: Adapting software to the specific requirements and needs of users
category:
- Requirements
- Business
problems:
- feature-gaps
- poor-user-experience-ux-design
- user-frustration
- customer-dissatisfaction
- negative-user-feedback
- vendor-lock-in
layout: solution
related_solutions:
- slug: adaptive-behavior
  similarity: 0.75
- slug: consistent-user-interface
  similarity: 0.75
- slug: customizable-user-interface
  similarity: 0.7
- slug: standard-software
  similarity: 0.7
- slug: a-b-testing
  similarity: 0.7
- slug: explicit-extension-points
  similarity: 0.7
---

## Description

Customizing adapts a system's behavior to the specific requirements of different user groups through configuration-driven mechanisms — feature flags, tenant-specific settings, plugin architectures, extension points — rather than through hard-coded, one-size-fits-all logic that treats every user the same way. Legacy systems are especially prone to this rigidity because their core workflows were typically designed around the needs of a single original user group, and as the system's user base diversified over time, groups whose workflows do not match that original design were left to build their own workarounds — spreadsheets, manual processes, shadow systems — outside the software rather than within it. Introducing extension points and configuration-driven behavior lets these divergent needs be met without modifying the core system for each one individually, and critically, without forking the codebase into parallel, separately maintained versions for each user group. Separating customization from core code also protects the investment during upgrades, since custom configuration that lives outside the core system is not at risk of being silently overwritten the next time the underlying software is updated. Left unchecked, however, customization has a tendency to accumulate its own technical debt, since every additional customization point multiplies the testing matrix and creates a new opportunity for configuration-related bugs and unexpected interactions between different customizations.

## How to Apply ◆

- Identify areas where the legacy system's one-size-fits-all approach fails specific user groups and prioritize customization efforts accordingly.
- Introduce configuration-driven behavior (feature flags, user preferences, tenant-specific settings) rather than hard-coded logic.
- Build extension points in the legacy system that allow user-specific behavior without modifying core code.
- Use plugin architectures or strategy patterns to make business rules customizable without code changes.
- Gather user feedback systematically to understand which customization options deliver the most value.
- Ensure customizations are maintainable across upgrades by separating custom code from the core system.

## Tradeoffs ⇄

**Benefits:**
- Increases user satisfaction by adapting the system to actual workflows rather than forcing users to adapt.
- Reduces the need for workarounds and shadow systems that users create when the software does not fit their needs.
- Enables the same legacy system to serve different user groups or tenants without forking.

**Costs:**
- Excessive customization can make the system harder to maintain, test, and upgrade.
- Each customization point increases the testing matrix and potential for configuration-related bugs.
- Can lead to feature bloat if customization requests are not prioritized carefully.
- Custom configurations can conflict with each other in unexpected ways.

## How It Could Be

A legacy CRM system serves both inside sales and field service teams, but its rigid workflow forces field service technicians to navigate screens designed for sales representatives. Rather than building a separate system, the team introduces role-based UI configurations and customizable workflow templates. Field service technicians see only the fields and steps relevant to their work, while sales representatives retain their current experience. The configuration is stored separately from core code, so system upgrades do not overwrite customizations. User satisfaction surveys show marked improvement for the field service team, and the workaround spreadsheets they previously maintained to compensate for the rigid UI are no longer needed.
