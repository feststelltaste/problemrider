---
title: User Stories
description: Formulate requirements from the user's perspective
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/user-stories
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- misaligned-deliverables
- stakeholder-developer-communication-gap
- feature-bloat
- large-feature-scope
- implementation-rework
layout: solution
---

## How to Apply ◆

> In legacy modernization, user stories shift the focus from replicating technical features to delivering user value, preventing the common trap of rebuilding everything "because the old system had it."

- Write user stories for the replacement system based on what users need to accomplish, not on what the legacy system's screens and functions look like.
- Use the format "As a [user role], I want [capability] so that [business value]" to force the team to articulate why each piece of functionality matters.
- Break down legacy system features into user stories that can be delivered and validated independently, enabling incremental migration rather than big-bang replacement.
- Include acceptance criteria on each story that define clear, testable conditions of satisfaction based on business outcomes.
- Involve users in story writing workshops to capture requirements that only exist as tacit knowledge in the legacy system.
- Prioritize stories based on user value and migration risk rather than technical convenience, ensuring that the most critical user needs are addressed first.
- Use story splitting techniques to keep stories small enough for single-iteration delivery while maintaining meaningful user value.

## Tradeoffs ⇄

> User stories keep development focused on user value but require ongoing refinement and can be challenging to write for complex legacy business logic.

**Benefits:**

- Prevents feature bloat during modernization by requiring explicit justification for each capability rather than blindly replicating legacy features.
- Enables incremental delivery and validation, allowing users to provide feedback on completed stories before the entire system is built.
- Creates a shared language between developers and stakeholders that focuses on outcomes rather than technical implementation details.
- Makes prioritization decisions transparent by connecting each story to a user need and business value.

**Costs and Risks:**

- Complex legacy business logic may be difficult to express as user stories without losing important nuances and edge cases.
- Stories written without sufficient domain understanding may miss critical legacy behavior that users take for granted.
- Over-splitting stories to fit sprint timeslots can fragment user workflows into pieces too small to validate meaningfully.
- Teams may write stories that are thinly disguised technical tasks rather than genuine user-value expressions.

## How It Could Be

> The following scenario demonstrates how user stories guide legacy modernization decisions.

A credit union was replacing its legacy loan origination system. The legacy system had 47 screens, and the initial plan was to rebuild each screen. When the team rewrote these as user stories from the loan officer's perspective, they discovered that 12 screens existed only to work around limitations of the legacy system's batch processing — they were used to manually trigger recalculations that the new system could perform automatically. By focusing on user stories rather than screen replication, the team eliminated 25% of the planned work while actually improving the loan officer's workflow. The story "As a loan officer, I want to see the updated monthly payment immediately when I change the interest rate, so that I can discuss options with the member in real time" replaced three legacy screens and a batch process with a single responsive calculation.
