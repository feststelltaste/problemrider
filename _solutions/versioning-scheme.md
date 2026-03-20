---
title: Versioning Scheme
description: Define when and why version numbers change to signal compatibility intent
category:
- Process
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/compatibility/versioning-scheme
problems:
- api-versioning-conflicts
- breaking-changes
- legacy-api-versioning-nightmare
- dependency-version-conflicts
- integration-difficulties
- change-management-chaos
layout: solution
---

## How to Apply ◆

- Choose a versioning scheme appropriate to the artifact type: semantic versioning for libraries, date-based versioning for data formats, or URI-based versioning for APIs.
- Document the versioning policy explicitly, defining what constitutes a breaking change, a feature addition, and a bug fix in the context of your legacy system.
- Apply the versioning scheme retroactively to legacy components by auditing their current state and assigning a baseline version.
- Integrate version validation into build and deployment pipelines to enforce the scheme.
- Communicate version changes through changelogs, release notes, and automated notifications to downstream consumers.
- Review the versioning scheme periodically to ensure it still serves the evolving system landscape.

## Tradeoffs ⇄

**Benefits:**
- Gives consumers a reliable signal about the nature and risk of an update.
- Enables automation of dependency management and upgrade decisions.
- Creates a common language for discussing change impact across teams.

**Costs:**
- Requires team discipline to categorize changes correctly and bump versions accordingly.
- Choosing the wrong scheme can create confusion rather than clarity.
- Retrofitting versions onto legacy components without prior versioning requires careful analysis.
- Different versioning schemes across the portfolio can reduce the intended clarity.

## How It Could Be

A large enterprise maintains over fifty internal libraries used by legacy applications. Historically, libraries were versioned with arbitrary build numbers that conveyed no compatibility information. The architecture team introduces a uniform versioning scheme: semver for libraries, calendar versioning for data schemas, and URL-path versioning for REST APIs. Each team documents their versioning policy in a shared wiki. Build pipelines enforce that pull requests include a version bump and a changelog entry. Within a few months, developers across the organization can assess upgrade risk at a glance, and automated tools handle patch-level updates without human intervention.
