---
title: Cross-Platform Frameworks
description: Utilize development frameworks that enable cross-platform applications
category:
- Architecture
- Code
problems:
- technology-lock-in
- vendor-lock-in
- difficult-code-reuse
- high-maintenance-costs
- duplicated-effort
- scaling-inefficiencies
layout: solution
related_solutions:
- slug: platform-independent-programming-languages
  similarity: 0.8
- slug: cross-platform-build-tools
  similarity: 0.8
- slug: platform-independence
  similarity: 0.75
- slug: platform-independent-data-storage
  similarity: 0.75
- slug: platform-independent-test-frameworks
  similarity: 0.75
- slug: emulation
  similarity: 0.75
---

## Description

Cross-platform frameworks such as Kotlin Multiplatform, Flutter, or .NET MAUI let a single codebase target multiple platforms, typically by sharing business logic across platforms while leaving genuinely platform-specific concerns — UI rendering, hardware access — implemented natively where needed. Organizations running separate native applications for each platform, maintained by separate teams, commonly discover that keeping feature parity between them is a permanent, losing battle: one platform's team ships faster than the other's, and the gap between the two versions widens with each release cycle rather than closing. Migrating the shared business logic — the parts that do not inherently depend on a specific platform, such as domain rules, scheduling logic, or offline synchronization — onto a cross-platform framework removes the duplicated implementation effort that caused the parity gap in the first place, without necessarily touching the platform-specific UI layers that benefit most from staying native. Because a full rewrite of two established native codebases at once is high-risk, this is typically approached as a gradual migration that starts with the most clearly separable, non-platform-specific logic and expands from there. The tradeoff is a new dependency on the framework's own roadmap and platform-feature coverage, some potential performance cost for UI- or hardware-intensive operations, and the reality that not every legacy codebase can be cleanly separated into shareable and platform-specific layers in the first place.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Evaluate cross-platform frameworks (React Native, Flutter, .NET MAUI, Kotlin Multiplatform, Electron) based on the application's requirements
- Identify the portion of legacy code that contains business logic separable from platform-specific UI or system code
- Start by porting shared business logic to the cross-platform framework while keeping platform-specific features native
- Use the framework's platform channel mechanisms for accessing native capabilities not covered by the framework
- Establish a testing strategy that covers both shared code and platform-specific adaptations
- Plan for a gradual migration rather than rewriting the entire application at once
- Monitor platform-specific performance to ensure the cross-platform layer does not introduce unacceptable overhead

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces development effort by sharing code across platforms instead of maintaining separate codebases
- Ensures consistent behavior and feature parity across platforms
- Enables smaller teams to support multiple platforms simultaneously
- Reduces time-to-market for features by implementing them once

**Costs and Risks:**
- Cross-platform frameworks may not support all native platform features or may lag behind platform updates
- Performance may be lower than fully native implementations for UI-intensive or hardware-intensive operations
- Creates dependency on the framework vendor's roadmap and support lifecycle
- Developers may need to learn framework-specific patterns in addition to platform knowledge
- Not all legacy codebases can be cleanly separated into shareable and platform-specific layers

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A field service company maintained separate legacy applications for iOS (Objective-C) and Android (Java), each with its own development team. Feature parity was a constant struggle, with the Android version typically running three months behind iOS. The team migrated shared business logic (work order management, scheduling, offline sync) to Kotlin Multiplatform while keeping the UI native. This reduced the codebase by 40 percent, eliminated the feature parity gap, and allowed one developer from each platform team to move to other projects. Critical platform-specific features like background GPS tracking remained native, ensuring no loss in functionality.
