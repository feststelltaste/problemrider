---
title: Feature Detection
description: Query system capabilities at runtime instead of relying on version numbers
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/compatibility/feature-detection
problems:
- technology-lock-in
- vendor-lock-in
- deployment-environment-inconsistencies
- inconsistent-behavior
- brittle-codebase
- hidden-dependencies
- dependency-version-conflicts
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify platform-specific conditionals in the codebase that use compile-time flags or version checks
- Replace version-based branching with runtime capability probes that test whether a feature or API is actually available
- Implement graceful fallbacks for each detected capability so the application degrades smoothly on less capable platforms
- Create an abstraction layer that encapsulates feature detection logic, keeping the rest of the codebase platform-agnostic
- Add logging when fallbacks are triggered so the team can track which environments lack expected capabilities
- Write tests that simulate both the presence and absence of platform features to verify fallback behavior

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates brittle version checks that break when platforms evolve or diverge
- Allows the application to run on a wider range of environments without code changes
- Provides graceful degradation instead of hard failures on unsupported platforms
- Makes the system more resilient to unexpected environment differences

**Costs and Risks:**
- Runtime detection adds overhead compared to compile-time decisions, though usually negligible
- Fallback code paths receive less testing and may hide subtle bugs
- Increased code complexity from maintaining multiple execution paths
- Some features cannot be meaningfully probed at runtime and still require conditional compilation

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy web application relied on browser user-agent strings to decide which JavaScript APIs to use, resulting in frequent breakages as new browser versions were released. The team replaced user-agent sniffing with Modernizr-style feature detection, probing for capabilities like WebSocket support and CSS Grid at runtime. When a feature was absent, the application fell back to polyfills or simpler alternatives. This eliminated the constant maintenance burden of updating browser version lists and reduced cross-browser defect reports by roughly 60%.
