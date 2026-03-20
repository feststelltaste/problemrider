---
title: Platform-Independent Programming Languages
description: Using programming languages that run on different systems without modifications
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/portability/platform-independent-programming-languages
problems:
- technology-lock-in
- vendor-lock-in
- vendor-dependency-entrapment
- obsolete-technologies
- legacy-skill-shortage
- stagnant-architecture
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Assess the current technology stack for platform-specific language dependencies such as native compiled code or OS-specific APIs
- Evaluate cross-platform languages (e.g., Java, Python, Go, Kotlin, C#/.NET) based on the project's performance, ecosystem, and team skill requirements
- Plan an incremental migration strategy starting with peripheral modules rather than core business logic
- Use interoperability mechanisms (FFI, REST APIs, message queues) to allow new cross-platform components to coexist with legacy platform-specific code
- Invest in team training for the target language before beginning large-scale migration
- Establish coding standards that avoid platform-specific idioms even within cross-platform languages

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates the need to maintain separate codebases for different target platforms
- Broadens the talent pool since cross-platform languages typically have larger developer communities
- Simplifies deployment across heterogeneous environments
- Reduces long-term maintenance costs by consolidating on a single portable codebase

**Costs and Risks:**
- Cross-platform languages may have lower performance than platform-native alternatives for compute-intensive tasks
- Language migration requires significant investment in rewriting and revalidation
- Some platform-specific features may not be accessible through cross-platform language abstractions
- Team productivity drops during the transition period while developers learn the new language

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A telecommunications company maintained a network monitoring system written in Delphi that only ran on Windows. As the company moved toward Linux-based infrastructure, the Delphi codebase became a bottleneck. The team chose Go for its cross-platform compilation, strong concurrency model, and single-binary deployment. They migrated module by module over twelve months, using REST APIs to connect the new Go services with remaining Delphi components. The final system compiled and ran identically on Windows and Linux, allowing gradual infrastructure migration without disrupting operations.
