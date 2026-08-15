---
title: Compatibility Matrix
description: Define supported combinations of configurations
category:
- Testing
- Operations
problems:
- deployment-environment-inconsistencies
- configuration-drift
- configuration-chaos
- integration-difficulties
- dependency-version-conflicts
- poor-system-environment
- abi-compatibility-issues
layout: solution
related_solutions:
- slug: documentation-of-compatibility-requirements
  similarity: 0.8
- slug: compatibility-testing
  similarity: 0.75
- slug: compatibility-requirements
  similarity: 0.75
- slug: cross-version-testing
  similarity: 0.75
- slug: compatibility-governance
  similarity: 0.7
- slug: requirements-traceability-matrix
  similarity: 0.7
---

## Description

A compatibility matrix is an explicit, documented statement of which combinations of operating systems, runtime versions, databases, browsers, or other environment variables a system officially supports, converting an implicit and often inconsistent assumption about "what should work" into a concrete, testable, and publishable specification. Once defined, the matrix drives what gets tested in CI, ensuring the configurations that matter most — the ones customers or the largest consumers actually run — receive coverage, while everything outside the matrix is explicitly out of scope for support. This is especially useful for legacy systems that have accumulated support for a wide, undocumented range of environments over many years without anyone ever writing down which combinations were actually verified to work, leaving both the support team and customers guessing whenever an issue is reported on an unfamiliar configuration. Publishing the matrix externally also lets consumers self-diagnose whether their environment is supported before they file a ticket, and gives the team a defensible basis for declining to investigate bug reports that fall outside the documented boundaries. Reviewing and updating the matrix at each release is what keeps it aligned with reality, letting the team deliberately retire aging, costly-to-test configurations instead of supporting them indefinitely out of inertia. The tradeoff is that testing every combination in the matrix consumes real CI time and infrastructure, so an overly broad matrix can become as impractical to maintain fully as having no matrix at all, and customers still running unsupported configurations may reasonably feel abandoned when support is formally withdrawn.

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
