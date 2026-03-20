---
title: Portability Checklists
description: Create checklists to check portability with different systems and platforms
category:
- Process
quality_tactics_url: https://qualitytactics.de/en/portability/portability-checklists
problems:
- technology-lock-in
- vendor-lock-in
- deployment-environment-inconsistencies
- quality-blind-spots
- inconsistent-quality
- poor-documentation
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Define a portability checklist covering key areas: OS dependencies, database compatibility, file system assumptions, network configuration, and external service integrations
- Include checks for platform-specific constructs such as path separators, line endings, character encodings, and endianness
- Integrate the checklist into code review processes so portability is verified before merging changes
- Review and update the checklist whenever a new target platform is added or a portability issue is discovered
- Automate checklist items where possible by adding linting rules or static analysis checks
- Use the checklist during architecture reviews and technology selection decisions to ensure new components meet portability requirements

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides a systematic approach to identifying portability risks before they reach production
- Serves as institutional knowledge that survives team member turnover
- Creates consistency in how portability is evaluated across teams and projects
- Low-cost practice that can be adopted immediately without tooling changes

**Costs and Risks:**
- Checklists can become stale if not regularly maintained and updated
- Checkbox compliance can become perfunctory without genuine investigation
- Manual checklists do not scale well for large codebases or frequent changes
- May create a false sense of security if important portability aspects are not covered

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A software consulting firm developed applications for clients running diverse infrastructure. After repeated portability issues during deployments, they created a portability checklist covering 40 items across six categories. The checklist was integrated into their pull request template so every change was evaluated against it. Over a year, the number of deployment-time portability failures dropped by 70%. The checklist also became a valuable onboarding tool, helping new developers understand the kinds of platform assumptions to avoid. When they later automated 25 of the 40 checks as CI pipeline rules, the remaining manual items became more focused and meaningful.
