---
title: Progressive Disclosure
description: Gradual disclosure of information and functions
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/progressive-disclosure/
problems:
- cognitive-overload
- increased-cognitive-load
- poor-user-experience-ux-design
- user-confusion
- user-frustration
- feature-bloat
- negative-user-feedback
- difficult-developer-onboarding
layout: solution
---

## How to Apply ◆

> Legacy systems tend to expose all functionality simultaneously, overwhelming users with options they rarely need. Progressive disclosure shows essential features first and reveals complexity only when needed.

- Identify the core actions that eighty percent of users perform eighty percent of the time. Make these prominently visible while moving advanced or rarely used features behind expandable sections, "Advanced" links, or secondary menus.
- Use expandable and collapsible sections for detailed information. Show summary data by default and let users expand to see full details when needed.
- Implement contextual menus that show relevant actions based on the current state of the data or workflow step, rather than displaying every possible action at all times.
- Layer form complexity: show basic fields by default and provide an "Advanced options" toggle for fields that only experienced users or unusual scenarios require.
- Use drill-down navigation for hierarchical data instead of showing all levels simultaneously. Let users navigate from summary to detail at their own pace.
- Apply progressive disclosure to settings and configuration pages, where legacy systems often present hundreds of options on a single screen.

## Tradeoffs ⇄

> Progressive disclosure simplifies the interface for most users but can frustrate power users who want immediate access to advanced features.

**Benefits:**

- Dramatically reduces cognitive overload by presenting only the information and actions relevant to the user's immediate task.
- Makes the system more approachable for new users who are not yet ready for advanced features.
- Reduces the visual clutter that makes legacy interfaces feel overwhelming and outdated.
- Allows the system to support both simple and complex use cases without requiring separate interfaces for different user levels.

**Costs and Risks:**

- Power users who frequently use advanced features may find progressive disclosure slows them down if they must click through extra steps to reach the functionality they need.
- Hiding features too aggressively can make them undiscoverable, causing users to believe functionality is missing when it is merely hidden.
- Implementing progressive disclosure in legacy frontends with rigid layouts may require significant restructuring of page templates and components.
- The line between essential and advanced features varies by user role, potentially requiring role-based progressive disclosure configurations.

## How It Could Be

> Legacy systems that grew feature-by-feature over the years often present every feature as equally important, creating an interface that serves no one well.

A legacy warehouse management system has a product editing screen with forty-two fields, including basic information like name and SKU, inventory parameters, supplier details, customs classification codes, and hazardous materials handling instructions. Warehouse clerks who need to update stock counts must scroll through all forty-two fields to find the inventory section. The team restructures the screen into a tabbed layout with "Basic Info" shown by default, and "Inventory," "Supplier," "Compliance," and "Advanced" tabs available for users who need them. Each tab contains only the relevant fields. The Basic Info tab covers the needs of eighty percent of daily editing tasks. Warehouse clerks report that the editing screen is no longer intimidating, and new employees can start making basic updates on their first day rather than requiring a week of training to understand the full form.
