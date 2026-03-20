---
title: Platform-Independent Build Pipelines
description: Implementing CI/CD pipelines that run on different build servers
category:
- Operations
- Process
quality_tactics_url: https://qualitytactics.de/en/portability/platform-independent-build-pipelines
problems:
- vendor-lock-in
- technology-lock-in
- complex-deployment-process
- manual-deployment-processes
- deployment-environment-inconsistencies
- long-build-and-test-times
- inefficient-development-environment
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Define build steps in a platform-agnostic format (e.g., Makefile, shell scripts, or containerized build images) that any CI server can invoke
- Avoid using CI-server-specific features or proprietary plugins for core build logic
- Use container-based build agents so the build environment is reproducible regardless of the CI platform
- Store pipeline definitions as code in the repository alongside the application source
- Abstract environment-specific variables through a configuration layer rather than embedding them in pipeline definitions
- Test the pipeline on at least two different CI platforms periodically to verify portability

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Avoids lock-in to a specific CI/CD vendor, making it easier to switch providers
- Ensures build reproducibility across developer machines, CI servers, and production environments
- Simplifies onboarding since developers can run the same build steps locally
- Reduces risk when a CI vendor changes pricing, features, or discontinues service

**Costs and Risks:**
- Restricting to platform-agnostic features means missing out on vendor-specific optimizations like native caching
- Maintaining portability across CI platforms adds testing and validation overhead
- Container-based builds may introduce additional startup latency compared to native agents
- Some advanced pipeline features like matrix builds or approval gates differ significantly across platforms

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A mid-sized software company had their entire build pipeline deeply embedded in Jenkins with over 200 Groovy-based pipeline scripts using Jenkins-specific plugins. When Jenkins maintenance became a significant operational burden, migrating to GitHub Actions seemed to require rewriting every pipeline. The team refactored by extracting core build logic into Makefiles and Docker-based build images, with thin CI-specific wrappers that simply invoked these portable steps. The migration to GitHub Actions took three weeks instead of the estimated three months, and they retained the ability to fall back to Jenkins if needed.
