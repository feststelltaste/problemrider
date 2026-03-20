---
title: Personas
description: Characterizing representative user types through fictional characters
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/personas
problems:
- poor-user-experience-ux-design
- inadequate-requirements-gathering
- requirements-ambiguity
- misaligned-deliverables
- customer-dissatisfaction
- user-frustration
- user-confusion
- feature-bloat
layout: solution
---

## How to Apply ◆

> In legacy modernization, personas help teams understand who actually uses the old system and what they need, rather than simply replicating every existing feature.

- Interview actual users of the legacy system to identify distinct user groups with different goals, technical skill levels, and usage patterns.
- Create three to five personas that represent the primary user types, giving each a name, role description, goals, frustrations with the current system, and key tasks they perform.
- Validate personas against usage analytics from the legacy system if available — log data often reveals user behavior patterns that interviews miss.
- Use personas during feature prioritization to determine which legacy features are critical for which user types, avoiding the trap of rebuilding everything "because it was there."
- Reference personas in design reviews and sprint planning to keep discussions focused on user needs rather than technical preferences.
- Update personas as the modernization progresses and new user feedback becomes available.

## Tradeoffs ⇄

> Personas provide a shared vocabulary for discussing user needs but can oversimplify if not grounded in real user research.

**Benefits:**

- Prevents feature bloat during modernization by providing clear criteria for what each user type actually needs versus what the legacy system happened to offer.
- Creates empathy for end users within the development team, especially when team members have never used the legacy system themselves.
- Helps prioritize modernization efforts by identifying which user groups are most affected by legacy system limitations.
- Provides a common reference point for resolving disagreements about feature scope and design decisions.

**Costs and Risks:**

- Poorly researched personas based on assumptions rather than real user data can lead the team astray, creating a false sense of understanding.
- Personas can become stale if not updated as the modernization changes user workflows and expectations.
- Teams may over-optimize for one persona at the expense of others if persona priorities are not well balanced.

## Examples

> The following scenario illustrates how personas guide legacy system modernization decisions.

A university replacing its legacy student registration system created four personas: a first-year student unfamiliar with the process, a senior student registering for the last time, an academic advisor managing hundreds of students, and a registrar administrator handling exceptions. The legacy system treated all these users identically, presenting them with the same complex interface. By designing the replacement around persona-specific workflows, the team was able to simplify the student-facing experience dramatically while preserving the power-user features that advisors and administrators relied on. Features used only by administrators were moved behind role-based access rather than cluttering every user's interface.
