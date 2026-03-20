---
title: Compatibility Matrix
description: Define supported combinations of configurations
category:
- Testing
- Operations
quality_tactics_url: https://qualitytactics.de/en/compatibility/compatibility-matrix
problems:
- deployment-environment-inconsistencies
- configuration-drift
- configuration-chaos
- integration-difficulties
- dependency-version-conflicts
- poor-system-environment
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Document all supported combinations of operating systems, runtime versions, databases, and browser versions in a matrix
- Prioritize testing the most common combinations and those used by your largest consumers
- Automate matrix-driven testing in CI so that each build validates key configuration combinations
- Review and update the matrix with each release to add new and retire unsupported configurations
- Make the matrix publicly available so consumers can verify their environment is supported
- Use the matrix to scope compatibility bug reports: issues outside the matrix are out of scope

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Sets clear expectations for what is and is not supported, reducing ambiguous bug reports
- Focuses testing effort on the configurations that matter most
- Helps teams make informed decisions about when to drop support for old platforms

**Costs and Risks:**
- Testing all matrix combinations can be expensive in CI time and infrastructure
- An overly large matrix may be impractical to fully test on every commit
- Consumers using unsupported configurations may feel abandoned
- The matrix requires ongoing maintenance to stay accurate

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A middleware vendor supported a legacy product across multiple Java versions, databases, and operating systems but had no documented compatibility matrix. Customers frequently reported issues on untested configurations, consuming support resources. After defining a formal matrix of 24 supported combinations and automating CI tests for each, the team reduced compatibility-related support tickets by 60% and was able to clearly communicate to customers which configurations were entering end-of-life.
