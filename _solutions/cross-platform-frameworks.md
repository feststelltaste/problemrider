---
title: Cross-Platform Frameworks
description: Utilize development frameworks that enable cross-platform applications
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/portability/cross-platform-frameworks
problems:
- technology-lock-in
- vendor-lock-in
- difficult-code-reuse
- high-maintenance-costs
- duplicated-effort
- scaling-inefficiencies
layout: solution
---

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
