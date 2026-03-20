---
title: Backward Compatible APIs
description: Evolving API contracts without breaking existing consumers
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/compatibility/backward-compatible-apis
problems:
- breaking-changes
- legacy-api-versioning-nightmare
- api-versioning-conflicts
- poor-interfaces-between-applications
- integration-difficulties
- fear-of-breaking-changes
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Add new fields and endpoints rather than modifying or removing existing ones
- Make new request fields optional with sensible defaults so existing clients do not need to send them
- Use tolerant readers: consumers should ignore unknown fields rather than failing on them
- Apply contract tests that validate old consumer expectations still hold after changes
- Avoid changing the semantic meaning of existing fields or status codes
- When a field must change type or meaning, introduce a new field and deprecate the old one

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables continuous API evolution without coordinated consumer releases
- Reduces integration failures and production incidents caused by breaking changes
- Builds consumer confidence and simplifies partner onboarding

**Costs and Risks:**
- APIs accumulate deprecated fields and endpoints, increasing cognitive load for new developers
- Tolerant reader patterns can hide real bugs in data exchange
- Maintaining backward-compatible behavior in business logic adds implementation complexity
- Eventually requires cleanup through a formal deprecation cycle

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An insurance platform needed to change its claim submission API to support a new document format. Instead of modifying the existing document field, the team added a new optional field for the structured format while keeping the original field functional. Existing consumers continued submitting claims unchanged, while new consumers could opt into the richer format. Consumer-reported errors dropped to zero during the transition, compared to three major incidents during a previous breaking API change.
