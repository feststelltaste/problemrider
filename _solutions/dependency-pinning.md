---
title: Dependency Pinning
description: Locking external dependency versions for reproducible, compatible builds
category:
- Operations
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/compatibility/dependency-pinning
problems:
- dependency-version-conflicts
- deployment-environment-inconsistencies
- configuration-drift
- breaking-changes
- deployment-risk
- increasing-brittleness
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Use lock files (package-lock.json, Gemfile.lock, poetry.lock) to pin exact versions of all transitive dependencies
- Commit lock files to version control so all developers and CI systems use identical dependency trees
- Pin base images and tool versions in container builds for reproducible builds
- Establish a regular cadence for reviewing and updating pinned versions rather than leaving them indefinitely
- Use dependency scanning tools to identify pinned versions with known vulnerabilities
- Document the rationale for any version pins that deviate from the latest available version

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Ensures builds are reproducible across environments and over time
- Prevents unexpected breakage from transitive dependency updates
- Makes it easier to diagnose issues by knowing exactly which versions are in use

**Costs and Risks:**
- Pinned dependencies can become stale, accumulating security vulnerabilities and missing bug fixes
- Updating a deeply pinned dependency tree can trigger cascading version conflicts
- Teams may use pinning as an excuse to avoid necessary dependency updates
- Different pinning strategies across teams can create inconsistency

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A web application team experienced intermittent CI failures that could not be reproduced locally. Investigation revealed that the CI server resolved a slightly different version of a transitive dependency than developers' machines did. After introducing strict dependency pinning with committed lock files and pinned CI tool versions, the build became fully reproducible. The team also scheduled monthly dependency update reviews, which caught two security vulnerabilities in pinned libraries before they were exploited.
