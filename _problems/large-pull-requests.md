---
title: Large Pull Requests
description: Pull requests are too large to review effectively, leading to superficial
  reviews and missed issues.
category:
- Code
- Communication
- Process
related_problems:
- slug: rushed-approvals
  similarity: 0.65
- slug: review-bottlenecks
  similarity: 0.6
- slug: large-feature-scope
  similarity: 0.6
- slug: reduced-code-submission-frequency
  similarity: 0.6
- slug: inadequate-code-reviews
  similarity: 0.6
- slug: excessive-class-size
  similarity: 0.55
solutions:
- code-review-process-reform
- trunk-based-development
layout: problem
---

## Description

Large pull requests occur when developers submit code changes that are too extensive or complex for reviewers to examine thoroughly within reasonable time constraints. These oversized changes make it practically impossible to conduct meaningful code reviews, as reviewers either skip the review entirely, perform only superficial checks, or approve changes without fully understanding their implications. Large pull requests defeat the primary purposes of code review: catching bugs, sharing knowledge, and maintaining code quality.

## Indicators ⟡
- Pull requests regularly contain hundreds or thousands of lines of changes
- Code reviews take an unusually long time or are approved very quickly without meaningful feedback
- Reviewers frequently comment "LGTM" (Looks Good To Me) without substantial review comments
- Developers avoid reviewing certain pull requests due to their size and complexity
- Multiple unrelated features or bug fixes are bundled together in single pull requests

## Symptoms ▲

- [Superficial Code Reviews](superficial-code-reviews.md)
<br/>  Reviewers faced with oversized pull requests resort to surface-level checks, missing important design and logic issues.
- [Rushed Approvals](rushed-approvals.md)
<br/>  Large PRs are approved without thorough review because reviewers lack time or energy to examine the full changeset.
- [Review Bottlenecks](review-bottlenecks.md)
<br/>  Large pull requests take much longer to review, creating bottlenecks that delay the entire development pipeline.
- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  Bugs slip through superficial reviews of large PRs and reach production, increasing the defect rate.
- [Increased Bug Count](increased-bug-count.md)
<br/>  When large pull requests bypass effective review, more defects are introduced into the codebase undetected.
## Causes ▼

- [Large Feature Scope](large-feature-scope.md)
<br/>  Features that are too large to be broken into incremental changes naturally produce oversized pull requests.
- [Long-Lived Feature Branches](long-lived-feature-branches.md)
<br/>  Branches that accumulate changes over long periods result in massive pull requests when finally submitted for review.
- [Reduced Code Submission Frequency](reduced-code-submission-frequency.md)
<br/>  When developers batch changes and submit infrequently, each submission contains many more changes than necessary.
## Detection Methods ○
- **Pull Request Size Metrics:** Track lines of code changed, files modified, and complexity metrics for pull requests
- **Review Time Analysis:** Monitor how long reviews take and correlate with pull request size
- **Review Quality Assessment:** Analyze the depth and quality of feedback provided on different sized pull requests
- **Approval Patterns:** Identify pull requests that are approved quickly relative to their size
- **Developer Feedback:** Ask team members about their experience reviewing large pull requests

## Examples

A developer works on implementing a new user authentication system for three weeks in isolation. When they finally submit the pull request, it contains 2,500 lines of new code across 45 files, including database schema changes, new API endpoints, frontend components, configuration updates, and documentation changes. The assigned reviewers look at the massive pull request and either provide minimal feedback ("looks good overall") or focus only on obvious issues like code formatting, missing obvious bugs and architectural problems. Due to the size, no reviewer has the time or energy to understand the complete authentication flow, verify that security requirements are met, or ensure the implementation follows established patterns. Several critical security vulnerabilities make it into production because they were buried within the large changeset. Another example involves a pull request that combines a major refactoring of the data access layer with three new features and bug fixes for two existing features. The 1,800-line pull request spans multiple business domains and requires expertise in different areas of the system. Reviewers focus on the parts they understand best and skip areas outside their expertise, resulting in integration issues and inconsistent code quality across the different changes.
