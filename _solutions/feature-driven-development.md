---
title: Feature Driven Development
description: Structuring and implementing software functionality in the form of features
category:
- Process
- Management
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/feature-driven-development
problems:
- slow-feature-development
- poor-planning
- unclear-goals-and-priorities
- large-feature-scope
- delayed-value-delivery
- planning-dysfunction
layout: solution
---

## How to Apply ◆

- Decompose legacy system improvements into client-valued features expressed as "<action> the <result> <by|for|of|to> a(n) <object>" (e.g., "Calculate the total premium for a policy renewal").
- Build a feature list that serves as the backlog for legacy modernization work, organized by business area.
- Assign feature ownership to individual developers who are responsible for designing and implementing each feature end-to-end.
- Plan work in two-week iterations where each iteration delivers a set of completed features.
- Track progress through feature completion percentages, giving stakeholders visibility into modernization progress.
- Use design-by-feature and build-by-feature phases to ensure each feature is properly designed before implementation.

## Tradeoffs ⇄

**Benefits:**
- Keeps modernization efforts focused on delivering tangible business value through completed features.
- Provides clear progress tracking that stakeholders can understand.
- Prevents scope creep by defining features with specific, measurable deliverables.
- Assigns clear ownership, reducing coordination overhead.

**Costs:**
- Cross-cutting concerns (security, performance, infrastructure) do not fit neatly into features.
- Feature decomposition requires understanding of both the business domain and the legacy system.
- Individual feature ownership can create knowledge silos if not combined with code reviews and knowledge sharing.
- Legacy modernization often involves foundational work that is not directly feature-visible.

## How It Could Be

A legacy warehouse management system modernization effort stalls because the team is working on broad, undefined tasks like "improve the inventory module." The team switches to feature-driven development, creating a feature list of 120 specific features organized by business capability. Each two-week iteration, the team selects features to implement, designs them briefly, and builds them to completion. Stakeholders can see that 45% of warehouse receiving features are modernized, 20% of put-away features, and 0% of cycle counting features. This visibility allows the business to reprioritize: cycle counting is more urgent than put-away, so the team shifts focus. Within six months, the most business-critical features are modernized while lower-priority areas remain on the legacy system, maximizing the value delivered within the available budget.
