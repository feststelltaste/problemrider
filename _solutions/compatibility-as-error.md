---
title: Compatibility as Error
description: Treat compatibility regressions as build-breaking defects, not as acceptable
  technical debt
category:
- Process
- Testing
problems:
- breaking-changes
- regression-bugs
- fear-of-breaking-changes
- quality-blind-spots
- quality-degradation
- insufficient-testing
layout: solution
related_solutions:
- slug: compatibility-measurement
  similarity: 0.85
- slug: compatibility-governance
  similarity: 0.85
- slug: compatibility-standards
  similarity: 0.85
- slug: compatibility-testing
  similarity: 0.85
- slug: backward-compatibility
  similarity: 0.8
- slug: compatibility-certification
  similarity: 0.8
---

## Description

Compatibility as error is the practice of treating any backward-incompatible change to an API, schema, or interface as a build-breaking defect that blocks a release, rather than as an acceptable tradeoff or a piece of technical debt to be addressed later. It is enforced by wiring automated compatibility checks — contract tests that compare a proposed change against the previous stable version's schema — directly into the CI pipeline, so a regression is caught and blocks the merge before it ever reaches consumers, with the same urgency normally reserved for a failing security scan. This reframing matters in legacy contexts because compatibility breakage there tends to be treated reactively: a breaking change ships, an integration partner's system fails, and the team scrambles to patch the fallout after the fact, repeating the same costly cycle release after release. Making compatibility failures release-blocking by policy converts that reactive posture into a proactive one, since the cost of a break is now paid immediately by the change's author, in the form of a failed build, rather than downstream by every integration consumer weeks or months later. It does not forbid intentional breaking changes outright, but routes them through an explicit approval gate that requires a stated migration plan, distinguishing deliberate, coordinated evolution from accidental regression. The obvious risk is that overly strict or poorly tuned checks generate false positives that erode trust in the gate and invite teams to route around it, so the compatibility test suite itself needs to be trustworthy enough to justify blocking a release on its result.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A payment processing company experienced quarterly incidents where API changes broke merchant integrations. The team added a contract-testing step to their CI pipeline that compared each pull request against the currently deployed API schema. Any incompatible change failed the build immediately. In the first year, the number of compatibility-related production incidents dropped from twelve to one, and that single incident was traced to a configuration error rather than a code change.
