---
title: Long Build and Test Times
description: A situation where it takes a long time to build and test a system.
category:
- Process
related_problems:
- slug: long-release-cycles
  similarity: 0.6
- slug: work-queue-buildup
  similarity: 0.6
- slug: extended-cycle-times
  similarity: 0.6
- slug: extended-research-time
  similarity: 0.55
- slug: maintenance-bottlenecks
  similarity: 0.55
- slug: increased-time-to-market
  similarity: 0.55
solutions:
- parallelization
- pipelining
- ci-cd-pipeline
- continuous-integration
- continuous-integration-and-delivery
layout: problem
---

## Description
Long build and test times are a situation where it takes a long time to build and test a system. This is a common problem in large, monolithic architectures, where the entire system must be built and tested at once. Long build and test times can lead to a slowdown in development velocity, and they can also be a major source of frustration for developers.

## Indicators ⟡
- It takes a long time to get feedback on a change.
- Developers are often blocked waiting for the build to finish.
- The build is often broken.
- Developers are not able to run the tests on their local machines.

## Symptoms ▲

- [Slow Development Velocity](slow-development-velocity.md)
<br/>  Developers waiting for builds and tests cannot iterate quickly, directly reducing the team's development throughput.
- [Reduced Code Submission Frequency](reduced-code-submission-frequency.md)
<br/>  When builds take too long, developers batch changes together to avoid waiting, submitting less frequently.
- [Long-Lived Feature Branches](long-lived-feature-branches.md)
<br/>  Slow feedback from builds discourages frequent integration, causing branches to live longer before merging.
- [Developer Frustration and Burnout](developer-frustration-and-burnout.md)
<br/>  Constantly waiting for slow builds is demoralizing and disrupts developer flow, leading to frustration.
- [Lower Code Quality](lower-code-quality.md)
<br/>  Developers skip running full test suites locally due to long times, leading to more defects reaching shared branches.
- [Context Switching Overhead](context-switching-overhead.md)
<br/>  While waiting for builds, developers switch to other tasks, losing mental context and reducing effectiveness.
## Causes ▼

- [Monolithic Architecture Constraints](monolithic-architecture-constraints.md)
<br/>  Monolithic systems require building and testing the entire application together, making build times grow with system size.
- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  Tightly coupled components prevent incremental builds and require full recompilation and testing for any change.
- [High Coupling and Low Cohesion](high-coupling-low-cohesion.md)
<br/>  Poor module boundaries mean changes cascade across the codebase, requiring extensive rebuilding and retesting.
- [Uncontrolled Codebase Growth](uncontrolled-codebase-growth.md)
<br/>  An ever-growing codebase without modularization naturally leads to longer compile and test execution times.
- [Inadequate Test Infrastructure](inadequate-test-infrastructure.md)
<br/>  Lack of proper test infrastructure such as parallel test execution or test caching makes test runs unnecessarily slow.
## Detection Methods ○
- **Build and Test Time Monitoring:** Monitor the build and test times to identify which parts of the build are the slowest.
- **Developer Surveys:** Ask developers if they feel like they are able to get fast feedback on their changes.
- **Build and Test Log Analysis:** Analyze the build and test logs to identify errors and warnings.

## Examples
A company has a large, monolithic e-commerce application. It takes over an hour to build and test the application. The developers are often blocked waiting for the build to finish. The build is often broken, and it can take hours to fix it. The developers are not able to run all the tests on their local machines, so they are not able to get a complete picture of the quality of their code. As a result, the development velocity is slow, and the code quality is poor.
