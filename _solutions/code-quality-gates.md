---
title: Code Quality Gates
description: Ensure code quality through standardized, automated checks
category:
- Process
- Code
quality_tactics_url: https://qualitytactics.de/en/maintainability/code-quality-gates
problems:
- lower-code-quality
- high-technical-debt
- quality-degradation
- inconsistent-quality
- insufficient-code-review
- high-bug-introduction-rate
- regression-bugs
- quality-blind-spots
layout: solution
---

## How to Apply ◆

> In legacy systems, quality gates prevent new code from making things worse — they are the minimum investment needed to stop the bleeding while modernization proceeds.

- Define a set of automated quality checks that all code changes must pass before being merged: static analysis, test coverage thresholds, complexity limits, and dependency checks.
- Integrate quality gates into the CI/CD pipeline so that they run automatically on every pull request, providing immediate feedback without requiring manual intervention.
- Start with lenient thresholds appropriate for the legacy codebase's current state and tighten them incrementally — a gate set too high on day one will be bypassed or disabled.
- Implement a coverage ratchet that requires new code to meet higher coverage standards than the legacy baseline, preventing coverage regression.
- Include security scanning (SAST/DAST) in quality gates to catch vulnerabilities before they reach production.
- Make quality gate results visible in pull requests so that reviewers can focus on design and logic rather than mechanical quality checks.
- Review and adjust gate criteria quarterly based on the team's experience — gates that produce too many false positives will be ignored.

## Tradeoffs ⇄

> Quality gates prevent quality degradation automatically but require calibration to avoid being either too permissive or too restrictive.

**Benefits:**

- Prevents the common legacy system pattern of new code being as poor as existing code because "that is how it is done here."
- Provides objective, consistent quality enforcement that does not depend on individual reviewer diligence.
- Frees code reviewers to focus on higher-level concerns by automating mechanical quality checks.
- Creates a measurable quality floor that improves over time as thresholds are tightened.
- Makes quality expectations explicit and transparent for all developers.

**Costs and Risks:**

- Gates that are too strict for a legacy codebase create friction and may be circumvented through workarounds or exceptions.
- False positives from static analysis tools can erode trust in the quality gate process.
- Quality gates measure what tools can detect but miss design quality, naming clarity, and architectural fitness.
- Maintaining quality gate infrastructure and tool configurations requires ongoing effort.

## Examples

> The following scenario demonstrates how quality gates halt quality degradation in a legacy system.

A SaaS company's legacy platform had no automated quality checks, and code reviews were inconsistent — some reviewers checked quality rigorously while others approved anything that compiled. Over five years, this led to a codebase where quality varied wildly between modules. The team introduced quality gates requiring: minimum 70% line coverage for changed files, no new critical or major static analysis issues, no TODO comments without linked tickets, and all dependencies on approved versions. Initial resistance was high, with 40% of pull requests failing gates in the first month. But within three months, failure rates dropped to 15% as developers internalized the standards. After a year, the team tightened coverage requirements to 80% for new code and added complexity thresholds. Production defect rates in new features dropped by 45% compared to the pre-gate baseline.
