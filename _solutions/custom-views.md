---
title: Custom Views
description: Allow users to create their own views and layouts
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/custom-views/
problems:
- poor-user-experience-ux-design
- user-frustration
- shadow-systems
- feature-gaps
- negative-user-feedback
- customer-dissatisfaction
- user-confusion
layout: solution
---

## How to Apply ◆

> Legacy systems typically provide a single fixed view for each data set, forcing all users to work with the same layout regardless of their role or task. Custom views let users tailor the interface to their needs.

- Allow users to select which columns are visible in data tables and in what order. Legacy systems often display every available column, overwhelming users who need only a subset.
- Implement saved views that users can name, store, and switch between. Different tasks require different data perspectives, and users should not need to reconfigure their view each time.
- Support filtering and sorting presets that can be saved as part of a custom view, so users can quickly access their most common data subsets.
- Provide default views for common roles that serve as starting points, so new users have a reasonable layout without needing to configure one from scratch.
- Allow administrators to create shared views for teams or departments, reducing the effort of individual configuration while still supporting customization.
- Persist view preferences server-side so users see their customized interface regardless of which device or browser they use.

## Tradeoffs ⇄

> Custom views give users control over their workspace, but add complexity to the frontend and data layer.

**Benefits:**

- Reduces shadow system creation because users who can tailor the official system to their needs have less motivation to export data to spreadsheets for custom analysis.
- Addresses diverse user needs without requiring the development team to build role-specific interfaces for every use case.
- Improves user satisfaction by giving users agency over their workspace rather than forcing a one-size-fits-all layout.
- Reduces cognitive overload by allowing users to hide information they do not need for their current task.

**Costs and Risks:**

- Custom view functionality adds complexity to the frontend code, especially in legacy systems with rigid rendering pipelines that were not designed for dynamic layouts.
- Supporting and debugging issues in highly customized views is more difficult because the development team cannot reproduce the exact configuration a user is seeing.
- Users may create views that omit important information and then miss critical data, requiring safeguards such as mandatory columns for certain roles.
- Persisting view configuration requires additional database storage and API endpoints that the legacy system may not have.

## Examples

> One-size-fits-all interfaces in legacy systems drive users to create shadow systems where they can see the data the way they need it.

A legacy inventory management system displays a table with thirty-two columns for every product, from SKU and description to warehouse location, supplier codes, customs classifications, and reorder points. Warehouse staff need only five of these columns to do their daily work, while procurement staff need a different set of twelve columns. Both groups have been exporting data to Excel to create their own views, leading to duplicated effort and stale data. The team adds column selection and saved view functionality to the product table. Warehouse staff create a "Pick List" view with just the columns they need, and procurement staff create a "Reorder Review" view. Spreadsheet exports drop dramatically, and both groups report spending less time finding information and more time acting on it.
