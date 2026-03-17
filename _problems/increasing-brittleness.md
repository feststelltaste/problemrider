---
title: Increasing Brittleness
description: Software systems become more fragile and prone to breaking over time,
  with small changes having unpredictable and widespread effects.
category:
- Architecture
- Code
- Management
related_problems:
- slug: brittle-codebase
  similarity: 0.7
- slug: quality-degradation
  similarity: 0.65
- slug: rapid-system-changes
  similarity: 0.6
- slug: tight-coupling-issues
  similarity: 0.6
- slug: unpredictable-system-behavior
  similarity: 0.6
- slug: increased-bug-count
  similarity: 0.6
layout: problem
---

## Description

Increasing brittleness occurs when software systems become progressively more fragile and unstable over time, where seemingly minor changes can cause unexpected failures or break unrelated functionality. This brittleness develops as technical debt accumulates, dependencies become more complex, and the system architecture degrades without proper maintenance. Brittle systems are difficult to modify safely and often exhibit unpredictable behavior.

## Indicators ⟡

- Small changes frequently cause unexpected failures in unrelated system areas
- The number of bugs increases even when new features aren't being added
- System behavior becomes increasingly unpredictable
- More time is spent debugging than developing new functionality
- Changes that worked in development fail in production for unclear reasons

## Symptoms ▲

- [Fear of Change](fear-of-change.md)
<br/>  When small changes cause unpredictable failures, developers become afraid to modify the system.
- [Unpredictable System Behavior](unpredictable-system-behavior.md)
<br/>  A brittle system exhibits unexpected behavior when changes are made, making outcomes hard to predict.
- [Increased Bug Count](increased-bug-count.md)
<br/>  Brittle systems generate more bugs as changes cascade through tightly coupled components.
- [Slow Feature Development](slow-feature-development.md)
<br/>  Developers must proceed cautiously in brittle systems, extensively testing every change, which slows development.
- [Constant Firefighting](constant-firefighting.md)
<br/>  Frequent unexpected failures from brittleness keep teams in reactive mode, responding to cascading issues.
## Causes ▼

- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  Tightly coupled components mean changes propagate unpredictably through the system, making it fragile.
- [High Technical Debt](high-technical-debt.md)
<br/>  Accumulated technical debt degrades system architecture over time, making the system progressively more fragile.
- [Accumulation of Workarounds](accumulation-of-workarounds.md)
<br/>  Workarounds create hidden dependencies and bypass designed interfaces, making the system prone to breaking.
- [Quality Blind Spots](insufficient-testing.md)
<br/>  Without comprehensive tests, regressions go undetected as changes are made, allowing brittleness to grow.
- [Lower Code Quality](lower-code-quality.md)
<br/>  Low-quality code with unclear contracts and poor structure becomes increasingly fragile as it evolves.
## Detection Methods ○

- **Failure Rate Tracking:** Monitor the frequency of system failures and their relationship to recent changes
- **Change Impact Analysis:** Assess how often changes affect unrelated system areas
- **Bug Trend Analysis:** Track bug reports over time, particularly regression bugs
- **System Stability Metrics:** Measure system uptime, error rates, and performance consistency
- **Change Risk Assessment:** Evaluate the perceived risk associated with making system modifications

## Examples

An e-commerce platform experiences a critical failure in its product recommendation engine after a seemingly unrelated change to the user authentication system. Investigation reveals that the authentication change modified a shared caching layer that the recommendation engine relied on, even though this dependency wasn't documented anywhere. This type of unexpected failure happens increasingly often - a database schema change breaks the reporting system, a UI update causes checkout failures, and a performance optimization triggers inventory tracking bugs. The development team spends more time investigating and fixing these cascade failures than implementing new features. Another example involves a financial trading system where adding a new data validation rule causes existing trades to fail processing due to subtle changes in data flow timing. The system has become so interconnected and fragile that any change risks triggering failures in distant parts of the system, making development extremely slow and risky.
