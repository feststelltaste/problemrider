---
title: Semantic Versioning
description: Communicate compatibility intent through structured version numbers
category:
- Dependencies
- Process
quality_tactics_url: https://qualitytactics.de/en/compatibility/semantic-versioning
problems:
- api-versioning-conflicts
- breaking-changes
- dependency-version-conflicts
- legacy-api-versioning-nightmare
- integration-difficulties
- ripple-effect-of-changes
layout: solution
---

## How to Apply ◆

- Adopt the MAJOR.MINOR.PATCH versioning scheme for all libraries, APIs, and shared components in the legacy system.
- Define clear rules: increment MAJOR for breaking changes, MINOR for backward-compatible additions, PATCH for backward-compatible fixes.
- Retrofit existing legacy components with semantic versions by auditing their current interfaces and establishing a baseline version.
- Integrate version bumping into the CI/CD pipeline so developers must declare the type of change they are making.
- Publish changelogs alongside version bumps to document what changed and why.
- Use dependency management tools that respect semantic version ranges to prevent accidental upgrades to incompatible versions.

## Tradeoffs ⇄

**Benefits:**
- Consumers can immediately understand whether an update is safe to adopt by inspecting the version number.
- Reduces surprise breakages when upgrading dependencies in a legacy ecosystem.
- Provides a shared vocabulary across teams for discussing the impact of changes.
- Enables automated dependency update tools to make safe upgrade decisions.

**Costs:**
- Requires discipline; incorrect version bumps undermine trust in the scheme.
- Legacy components with no prior versioning need a potentially time-consuming audit to establish a baseline.
- Does not prevent breaking changes, only communicates them; enforcement requires additional tooling.
- Pre-1.0 versions have weaker guarantees, which can cause confusion during early modernization phases.

## How It Could Be

A legacy monolith is being decomposed into shared libraries consumed by multiple teams. Without semantic versioning, teams frequently pull in updates that silently break their builds. After adopting semver, each library publishes a changelog and increments its major version when APIs change. Consuming teams configure their dependency managers to accept only compatible minor and patch updates automatically. Major version bumps trigger a planned migration effort. Over six months, unplanned breakages from library updates drop substantially, and teams gain confidence in upgrading dependencies because the version number reliably signals intent.
