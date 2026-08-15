---
title: Feature Detection
description: Query system capabilities at runtime instead of relying on version numbers
category:
- Code
- Architecture
problems:
- technology-lock-in
- vendor-lock-in
- deployment-environment-inconsistencies
- inconsistent-behavior
- brittle-codebase
- hidden-dependencies
- dependency-version-conflicts
layout: solution
related_solutions:
- slug: cross-version-testing
  similarity: 0.7
- slug: compatibility-testing
  similarity: 0.7
- slug: documentation-of-compatibility-requirements
  similarity: 0.7
- slug: feature-toggles
  similarity: 0.7
- slug: forward-compatibility
  similarity: 0.7
- slug: compatibility-as-error
  similarity: 0.65
---

## Description

Feature detection queries a runtime environment for whether a specific capability is actually present — testing for a concrete API or behavior directly — rather than branching on a version number or identifier and assuming what that version implies about available functionality. Legacy codebases that branch on version strings, such as browser user-agent sniffing or OS version checks, are fragile in a specific way: the assumed correlation between a version number and a capability breaks the moment a new version changes what it supports, or a previously reliable identifier gets spoofed or deprecated, and every such break requires another round of manual updates to the version-matching logic. Replacing these checks with direct capability probes, encapsulated behind an abstraction layer with a graceful fallback for every detected absence, removes that maintenance burden entirely and lets the same code run correctly across a wider and less predictable range of environments, degrading smoothly rather than failing outright where a capability is missing. The cost is a small amount of runtime overhead for the probes themselves, less exercised fallback code paths that can hide their own latent bugs, and added branching complexity from maintaining multiple execution paths side by side.

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
