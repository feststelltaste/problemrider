---
title: Compatibility as Error
description: Treat compatibility regressions as build-breaking defects, not as acceptable technical debt
category:
- Process
- Testing
quality_tactics_url: https://qualitytactics.de/en/compatibility/compatibility-as-error
problems:
- breaking-changes
- regression-bugs
- fear-of-breaking-changes
- quality-blind-spots
- quality-degradation
- insufficient-testing
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Add compatibility checks to the CI pipeline that fail the build on any backward-incompatible change
- Use contract testing tools to automatically detect API or schema regressions
- Define compatibility as a release-blocking criterion in your definition of done
- Create automated compatibility test suites that run against the previous stable version
- Treat compatibility failures with the same urgency as security vulnerabilities: fix before merge
- Establish a review gate where intentional breaking changes require explicit approval and a migration plan

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Catches breaking changes before they reach consumers, preventing production incidents
- Shifts the team mindset from reactive compatibility fixes to proactive compatibility assurance
- Reduces the total cost of integration failures across the organization

**Costs and Risks:**
- Can slow down development when the pipeline blocks on compatibility checks
- Requires investment in tooling and test infrastructure for compatibility validation
- Overly strict rules may frustrate teams that need to make intentional breaking changes
- False positives in compatibility checks can erode trust in the process

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A payment processing company experienced quarterly incidents where API changes broke merchant integrations. The team added a contract-testing step to their CI pipeline that compared each pull request against the currently deployed API schema. Any incompatible change failed the build immediately. In the first year, the number of compatibility-related production incidents dropped from twelve to one, and that single incident was traced to a configuration error rather than a code change.
