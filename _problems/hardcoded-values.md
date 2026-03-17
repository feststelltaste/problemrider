---
title: Hardcoded Values
description: Magic numbers and fixed strings reduce flexibility, making configuration
  and adaptation difficult
category:
- Architecture
- Code
related_problems:
- slug: legacy-configuration-management-chaos
  similarity: 0.55
- slug: brittle-codebase
  similarity: 0.5
layout: problem
---

## Description

Hardcoded values are literal numbers, strings, or other constants embedded directly in source code rather than being defined as configurable parameters, constants, or external configuration. This practice reduces system flexibility by making it difficult to modify behavior without changing and redeploying code. The problem is particularly problematic in systems that need to adapt to different environments, handle varying business rules, or accommodate changing requirements over time.

## Indicators ⟡

- Code that contains unexplained numeric literals or "magic numbers" without context
- String values like URLs, file paths, or messages embedded directly in business logic
- Different versions of similar code that vary only by hardcoded values
- Requests for "simple" configuration changes that require code modifications
- Difficulty setting up the same application in different environments
- Business rules that are scattered throughout the codebase as literal values
- Test files that duplicate production code just to change embedded values

## Symptoms ▲

- [Deployment Environment Inconsistencies](deployment-environment-inconsistencies.md)
<br/>  Hardcoded values that are correct for one environment cause failures when the application is deployed to different environments.
- [Slow Feature Development](slow-feature-development.md)
<br/>  Simple configuration changes require code modifications, testing, and redeployment instead of just updating a configuration file.
- [Brittle Codebase](brittle-codebase.md)
<br/>  Code with hardcoded values breaks easily when business rules, URLs, or other parameters change.
- [Code Duplication](code-duplication.md)
<br/>  Different versions of similar code that vary only by hardcoded values create duplicated logic throughout the codebase.
- [Difficult Code Reuse](difficult-code-reuse.md)
<br/>  Code with embedded literal values cannot be easily reused in different contexts or configurations.
- [Increased Cost of Development](increased-cost-of-development.md)
<br/>  What should be simple configuration changes become multi-week development projects requiring code changes and full testing.

## Causes ▼
- [Deadline Pressure](deadline-pressure.md)
<br/>  Time pressure leads developers to embed values directly in code as the quickest path to a working solution.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers lacking experience with configuration management patterns default to hardcoding values directly in source code.
- [Poor Planning](poor-planning.md)
<br/>  Failure to anticipate future configuration needs leads to values being embedded in code rather than externalized.

## Detection Methods ○

- Use static analysis tools to identify magic numbers and repeated string literals
- Code reviews that specifically look for unexplained literal values in business logic
- Analyze deployment processes to identify values that need to change between environments
- Review configuration change requests to identify patterns of hardcoded dependencies
- Examine test code for workarounds needed due to inflexible hardcoded values
- Survey operations and business teams about limitations in system configuration
- Audit codebase for repeated literal values that should be centralized as constants
- Monitor development time spent on changes that should be simple configuration updates

## Examples

An e-commerce application has shipping cost calculations hardcoded throughout the codebase with values like `if (weight > 50) shippingCost = 15.99` and timeout values like `setTimeout(checkStatus, 30000)`. When the business wants to offer promotions, adjust shipping rates for different regions, or optimize performance by changing timeout values, each change requires code modifications, testing, and deployment. A particularly problematic situation arises when they need to support international customers - the hardcoded USD currency symbols, US zip code validation patterns, and English error messages are scattered across dozens of files. What should be simple business configuration changes become multi-week development projects, and supporting multiple markets requires maintaining separate code branches with different hardcoded values.
