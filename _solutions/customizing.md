---
title: Customizing
description: Adapting software to the specific requirements and needs of users
category:
- Requirements
- Business
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/customizing
problems:
- feature-gaps
- poor-user-experience-ux-design
- user-frustration
- customer-dissatisfaction
- negative-user-feedback
- vendor-lock-in
layout: solution
---

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

## Examples

A legacy CRM system serves both inside sales and field service teams, but its rigid workflow forces field service technicians to navigate screens designed for sales representatives. Rather than building a separate system, the team introduces role-based UI configurations and customizable workflow templates. Field service technicians see only the fields and steps relevant to their work, while sales representatives retain their current experience. The configuration is stored separately from core code, so system upgrades do not overwrite customizations. User satisfaction surveys show marked improvement for the field service team, and the workaround spreadsheets they previously maintained to compensate for the rigid UI are no longer needed.
