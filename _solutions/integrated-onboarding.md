---
title: Integrated Onboarding
description: Support for new users in getting started with the software through tutorials, guides, or interactive tours
category:
- Communication
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/integrated-onboarding/
problems:
- inadequate-onboarding
- user-confusion
- user-frustration
- difficult-developer-onboarding
- poor-user-experience-ux-design
- increased-customer-support-load
- negative-user-feedback
- new-hire-frustration
layout: solution
---

## How to Apply ◆

> Legacy systems are notoriously difficult for new users to learn because the interface conventions are outdated and tribal knowledge is required to operate them effectively. Integrated onboarding smooths the learning curve.

- Implement a first-time user experience that activates on the user's first login, highlighting the most important interface elements and explaining the basic workflow through tooltips, popovers, or a guided tour.
- Break onboarding into task-based segments rather than attempting to teach the entire system at once. Guide users through their first key action, such as creating their first record or completing their first transaction.
- Allow users to replay the onboarding tour at any time through a help menu option. Users who dismiss the initial tour may want guidance later when they encounter unfamiliar areas of the system.
- Track onboarding completion and adapt the experience based on what the user has already learned. Do not re-explain concepts the user has already demonstrated they understand.
- Provide role-based onboarding paths that focus on the features relevant to each user's responsibilities rather than a generic tour of the entire system.
- Combine in-application onboarding with quick-start documentation that users can reference outside the application for more detailed explanations.

## Tradeoffs ⇄

> Integrated onboarding reduces the time-to-productivity for new users but requires investment in content creation and maintenance.

**Benefits:**

- Dramatically reduces the time new users need to become productive, which is especially valuable for legacy systems with steep learning curves.
- Decreases the support burden during onboarding by providing self-service guidance directly within the application.
- Reduces new hire frustration caused by being dropped into an unfamiliar legacy system with no guidance.
- Creates a consistent onboarding experience that does not depend on the availability or teaching skill of experienced colleagues.

**Costs and Risks:**

- Onboarding content must be updated whenever the interface changes, or it will point to elements that no longer exist or behave differently.
- Intrusive onboarding that interrupts experienced users or cannot be easily dismissed creates frustration rather than reducing it.
- Building interactive tours requires frontend development effort that competes with other priorities, especially in legacy systems with constrained development budgets.
- Onboarding that covers too much at once overwhelms new users rather than helping them, requiring careful scoping of what to include.

## How It Could Be

> New users of legacy systems often describe their first experience as being dropped into the cockpit of an airplane with no training.

A legacy project portfolio management system is used across a large organization, with new project managers joining several times per year. Previously, each new user required two days of one-on-one training with an experienced colleague who would walk them through the system's non-obvious navigation and terminology. The team implements an interactive onboarding tour that guides new users through creating their first project, adding team members, and setting up their first milestone. The tour uses highlighted tooltips that point to each relevant interface element and explain what it does in plain language. After deploying the onboarding feature, the formal training requirement drops from two days to a half-day session covering advanced topics, and new project managers report feeling confident using the basic features within their first week.
