---
title: Adapter
description: Translate between incompatible interfaces through an intermediary layer
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/compatibility/adapter
problems:
- poor-interfaces-between-applications
- integration-difficulties
- architectural-mismatch
- legacy-api-versioning-nightmare
- technology-stack-fragmentation
- breaking-changes
- vendor-dependency
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify integration points where legacy interfaces do not match what consuming code expects
- Create adapter classes or modules that implement the target interface and delegate to the legacy component
- Keep the adapter thin, performing only structural translation without adding business logic
- Use adapters to wrap third-party libraries so your codebase depends on your own interface, not the vendor's
- Introduce adapters incrementally at the most painful integration boundaries first
- Write tests that verify the adapter correctly translates between both interface contracts

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Allows legacy components to participate in modern architectures without rewriting them
- Isolates breaking changes from external systems to a single translation point
- Enables parallel development: teams can code against the target interface while the adapter bridges the gap

**Costs and Risks:**
- Each adapter adds a maintenance surface that must be kept in sync with both sides
- Adapters can mask deeper design problems, delaying necessary refactoring
- Poorly designed adapters may introduce subtle data-loss or semantic mismatches
- Proliferation of adapters can create its own complexity if not governed

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A financial services firm needed to integrate a 15-year-old COBOL-based account management system with a new REST-based customer portal. Rather than rewriting the COBOL system, the team built a set of adapters that translated REST calls into the COBOL copybook format and mapped responses back to JSON. This allowed the new portal to launch on schedule while the legacy system continued operating unchanged, and it gave the team a clear seam for future incremental replacement.
