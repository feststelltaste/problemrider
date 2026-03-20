---
title: Rushed Approvals
description: Pull requests are approved quickly without thorough examination due to
  time pressure or process issues.
category:
- Code
- Culture
- Process
related_problems:
- slug: inadequate-code-reviews
  similarity: 0.65
- slug: review-process-avoidance
  similarity: 0.65
- slug: large-pull-requests
  similarity: 0.65
- slug: review-bottlenecks
  similarity: 0.65
- slug: code-review-inefficiency
  similarity: 0.65
- slug: approval-dependencies
  similarity: 0.6
solutions:
- code-review-process-reform
- checklists
layout: problem
---

## Description

Rushed approvals occur when code reviews are completed hastily without adequate examination of the changes, often due to time pressure, process dysfunction, or cultural issues that prioritize speed over quality. These superficial reviews fail to catch bugs, miss opportunities for knowledge sharing, and allow poor design decisions to accumulate in the codebase. Rushed approvals defeat the primary purposes of code review and can be more harmful than no review at all because they create false confidence in code quality.

## Indicators ⟡
- Pull requests are approved within minutes of submission regardless of size or complexity
- Review comments are minimal or generic ("LGTM", "Ship it") without specific feedback
- Reviews focus only on obvious syntax issues while missing logic or design problems
- Reviewers approve changes in areas of code they're not familiar with
- Review approval times are consistently short across all types of changes

## Symptoms ▲

- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  Hastily approved code bypasses quality scrutiny, allowing more bugs and design flaws to reach production.
- [Review Process Breakdown](review-process-breakdown.md)
<br/>  Widespread rushed approvals undermine the entire review process, making it ineffective as a quality gate.
- [Regression Bugs](regression-bugs.md)
<br/>  Reviewers who don't carefully examine changes miss regressions that break existing functionality.
- [Inconsistent Coding Standards](inconsistent-coding-standards.md)
<br/>  Quick approvals skip enforcement of coding standards, allowing inconsistent patterns to enter the codebase.
- [High Technical Debt](high-technical-debt.md)
<br/>  Without thorough review, design shortcuts and poor patterns accumulate in the codebase as technical debt.
## Causes ▼

- [Time Pressure](time-pressure.md)
<br/>  Deadline pressure forces reviewers to prioritize speed over thoroughness, leading to superficial approvals.
- [Review Bottlenecks](review-bottlenecks.md)
<br/>  When a large backlog of pull requests creates pressure, reviewers rush through approvals to clear the queue.
- [Large Pull Requests](large-pull-requests.md)
<br/>  Overwhelmingly large pull requests discourage thorough review, leading reviewers to skim and approve rather than invest the significant time needed.
- [Reviewer Inexperience](reviewer-inexperience.md)
<br/>  Inexperienced reviewers who cannot identify real issues default to quick approval rather than admitting they don't understand the code.
## Detection Methods ○
- **Review Time Analysis:** Track how long reviewers spend examining code relative to change complexity
- **Review Comment Quality:** Analyze the depth and specificity of review feedback
- **Bug Correlation:** Compare bug rates in rushed reviews versus thorough reviews
- **Review Coverage:** Assess whether reviewers examine all changed files and understand the changes
- **Developer Feedback:** Survey team members about review thoroughness and quality

## Examples

A development team is under pressure to release a major feature before a competitor's product launch. Pull requests that normally would require 30-60 minutes of careful review are being approved in 2-3 minutes with comments like "looks good" or "LGTM." A complex pull request implementing new payment processing logic is approved by three reviewers within 5 minutes, despite containing subtle bugs in error handling and edge case management. None of the reviewers took time to understand the payment flow or verify that the implementation correctly handles all the business requirements. The rushed approval allows critical payment bugs to reach production, causing transaction failures and customer complaints that could have been prevented with proper review. Another example involves a security-sensitive authentication module where rushed reviews miss a SQL injection vulnerability because reviewers only glance at the code without tracing the data flow or understanding the security implications. The vulnerability is discovered months later during a security audit, requiring emergency patches and exposing the system to potential attacks that could have been prevented by thorough code review.
