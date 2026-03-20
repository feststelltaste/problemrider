---
title: Empty States and First-Use Guidance
description: Designing meaningful empty states with clear guidance on what to do next
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/empty-states-and-first-use-guidance/
problems:
- user-confusion
- poor-user-experience-ux-design
- user-frustration
- difficult-developer-onboarding
- negative-user-feedback
- inadequate-onboarding
- feature-gaps
layout: solution
---

## How to Apply ◆

> Legacy systems often display blank screens or empty tables with no explanation when there is no data to show, leaving users confused about whether the system is broken or they need to take action.

- Identify all screens in the legacy system that can appear empty, including lists, dashboards, search results, and detail views for new accounts or projects.
- Replace blank screens with informative empty states that explain why there is no data and what the user can do next. Include a clear call to action such as "Create your first project" or "Import data to get started."
- Design first-use experiences for new users or new feature areas that guide users through the initial setup steps rather than dropping them into an empty interface.
- Use illustrations or icons in empty states to make them visually distinct from error states. Users should immediately understand that the absence of data is expected, not a malfunction.
- Provide sample or demo data that new users can explore to understand the system before committing their own data. This is especially valuable in complex legacy systems with steep learning curves.
- Test empty states with actual new users to verify that the guidance is sufficient and the calls to action are clear.

## Tradeoffs ⇄

> Meaningful empty states turn potentially confusing moments into opportunities for user education, but require design and content effort.

**Benefits:**

- Eliminates user confusion when encountering screens with no data, which is a common source of support requests in legacy systems.
- Improves onboarding by guiding new users through their first actions rather than leaving them to figure out the system on their own.
- Reduces the perceived complexity of the system by providing clear entry points for getting started.
- Prevents users from assuming the system is broken or that they lack access when they see empty screens.

**Costs and Risks:**

- Designing and implementing empty states for every possible empty screen in a large legacy system requires content writing and design effort.
- Empty state content must be maintained as the system evolves; outdated guidance that refers to removed features is confusing.
- First-use guidance may become annoying for users who create new projects or accounts frequently, requiring a mechanism to skip or dismiss.
- Localizing empty state content into multiple languages adds to the translation workload.

## Examples

> The first impression of a legacy system for new users is often a screen with no data and no guidance, setting a negative tone from the start.

A legacy project management system presents new users with a completely blank dashboard after their first login. The sidebar contains cryptic menu labels like "WBS," "CR Log," and "Baseline Control." New project managers who are unfamiliar with the system's terminology open support tickets asking how to start using the system. The team redesigns the empty dashboard to display a welcome message with three clear steps: "Create your first project," "Invite team members," and "Set up your first milestone." Each step links directly to the relevant screen with a brief explanation of what it does. The team also adds helpful empty states to the project list, task board, and document repository screens. New user support tickets related to initial setup decrease substantially, and the average time from account creation to first meaningful action drops from days to hours.
