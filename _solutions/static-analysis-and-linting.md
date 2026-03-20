---
title: Static Code Analysis
description: Automated review of source code for potential issues and improvement opportunities
category:
- Code
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/static-code-analysis/
problems:
- inconsistent-coding-standards
- inconsistent-naming-conventions
- poor-naming-conventions
- mixed-coding-styles
- undefined-code-style-guidelines
- inconsistent-codebase
- hardcoded-values
- null-pointer-dereferences
- integer-overflow-and-underflow
- unreleased-resources
- style-arguments-in-code-reviews
- automated-tooling-ineffectiveness
layout: solution
---

## How to Apply ◆

> In a legacy codebase, static analysis is one of the few tools that can scan the full extent of years of accumulated problems without requiring anyone to understand the code first.

- Start with a "new code only" mode that enforces rules on changed lines while suppressing existing violations — this prevents the codebase from getting worse without overwhelming the team with thousands of pre-existing findings.
- Run a full baseline scan on the legacy codebase to generate an inventory of violations, then treat that inventory as a debt backlog rather than a to-do list requiring immediate action.
- Prioritize security-focused rules first (SQL injection patterns, hardcoded credentials, unescaped inputs) because legacy systems often pre-date modern security awareness and carry undetected vulnerabilities.
- Integrate analysis into the CI pipeline so that every pull request is gated on not introducing new violations — even if existing violations remain, the count must not grow.
- Use complexity metrics (cyclomatic complexity, cognitive complexity) to identify the modules where change risk is highest; these hotspots are where legacy incidents most frequently originate.
- Enable dead code detection to find unreferenced functions and classes that accumulated as features were replaced over decades — removing dead code reduces the surface area that developers must understand.
- Enforce architectural boundary rules using tools such as ArchUnit or Dependency Cruiser to detect and prevent further violation of the module structure that the original architects intended.
- Schedule periodic full-codebase scans (nightly or weekly) to catch violations in code that predates the current standards and to track whether the overall debt level is trending up or down.

## Tradeoffs ⇄

> Static analysis provides objective, scalable quality measurement for legacy systems, but its value depends entirely on the team's willingness to act on findings rather than suppress or ignore them.

**Benefits:**

- Surfaces quality problems across the entire legacy codebase systematically, including in modules that no current team member has read or touched.
- Provides objective, quantified metrics (technical debt ratio, violation counts, complexity scores) that make it possible to argue for remediation investment to non-technical stakeholders.
- Catches recurring error patterns — null dereferences, resource leaks, deprecated API usage — that appear frequently in legacy code written before current practices were established.
- Frees human reviewers from mechanical checks so they can focus on the design and business logic issues that automated tools cannot assess.
- Creates a quality baseline that makes it possible to measure whether modernization efforts are actually improving the codebase or just rearranging problems.

**Costs and Risks:**

- Legacy codebases typically generate thousands of violations on first scan; without a triage strategy, the volume is paralyzing and teams may disable the tool rather than address findings.
- False positives in legacy code are more common because the code often uses patterns that predate the conventions the tools expect, requiring significant rule customization to reduce noise.
- Analysis tools add time to CI pipelines; in legacy systems with slow build processes, adding another step to an already slow pipeline may create developer resistance.
- Teams that optimize for analysis scores rather than actual quality will suppress findings or restructure code to satisfy the metric without improving the underlying behavior.
- Static analysis cannot detect business logic errors, incorrect domain assumptions, or the kind of design-level debt (wrong abstractions, inappropriate coupling) that dominates many legacy systems.

## Examples

> The following scenarios illustrate how static analysis is introduced and used in real legacy modernization efforts.

A logistics company inherited a ten-year-old PHP application that had been extended by successive development agencies with no consistent coding standards. When the new internal team ran a static analysis scan for the first time, it produced over 8,000 findings. Rather than attempting to fix them all, the team categorized findings by severity and froze the existing count — any new code had to leave the violation count unchanged or lower it. After six months of steady incremental cleanup during regular feature work, the count had dropped to under 3,000, and the team had eliminated every critical security finding. The metrics gave management a concrete way to see progress without requiring them to read code.

An insurance company's claims processing system was written in Java and had not received architectural attention in years. The team added ArchUnit tests that encoded the intended layering rules — services must not directly access the database layer, domain objects must not reference infrastructure classes — and ran them in CI. On the first run, 47 violations appeared, most of them in modules that had grown organically beyond their original boundaries. The team used this list to prioritize their refactoring backlog, working through the violations in order of the modules most frequently changed during feature development.

A telecommunications provider wanted to modernize a C++ billing engine but had no idea which parts of the codebase were still actively executed. Dead code detection revealed that roughly 30% of the functions in the billing module were never called from any reachable entry point — they were remnants of billing models that had been retired years earlier. Removing this code reduced the maintainable surface area significantly and made the remaining logic easier to understand and test, which in turn reduced the estimated cost of the planned rewrite.
