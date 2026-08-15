---
title: Adapter
description: Translate between incompatible interfaces through an intermediary layer
category:
- Architecture
- Code
problems:
- poor-interfaces-between-applications
- integration-difficulties
- architectural-mismatch
- legacy-api-versioning-nightmare
- technology-stack-fragmentation
- breaking-changes
- vendor-dependency
- dependency-on-supplier
layout: solution
related_solutions:
- slug: abstraction-layers
  similarity: 0.8
- slug: facades
  similarity: 0.8
- slug: protocol-abstraction
  similarity: 0.8
- slug: api-gateway
  similarity: 0.8
- slug: dependency-injection
  similarity: 0.75
- slug: mediator
  similarity: 0.75
---

## Description

The Adapter pattern introduces a thin translation class or module that implements the interface expected by consuming code while internally delegating to a component whose existing interface does not match — converting calls, parameters, and return values between the two shapes without adding any business logic of its own. It is one of the most direct tools for integrating a legacy component into a newer architecture, because it lets the legacy side remain completely untouched while giving the rest of the system a clean, purpose-built interface to depend on. This is especially valuable in legacy modernization when a component's original interface was designed for a technology or protocol that no longer matches how the rest of the system communicates — a mainframe using fixed-width copybook records, a SOAP service in a REST-oriented landscape, or a third-party library whose API design does not fit the application's own conventions. By wrapping such a dependency behind an adapter that exposes the interface the application actually wants, breaking changes and vendor-specific quirks are absorbed at a single, well-defined translation point instead of leaking throughout the codebase. Adapters also enable parallel development, since a team can build against the target interface immediately while the adapter is developed independently to bridge the gap to the legacy side. Because an adapter only translates structure, it must be kept simple and easily testable in isolation; letting business rules creep into the translation layer, or accumulating too many undisciplined adapters, recreates the very coupling problems the pattern was meant to solve.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A financial services firm needed to integrate a 15-year-old COBOL-based account management system with a new REST-based customer portal. Rather than rewriting the COBOL system, the team built a set of adapters that translated REST calls into the COBOL copybook format and mapped responses back to JSON. This allowed the new portal to launch on schedule while the legacy system continued operating unchanged, and it gave the team a clear seam for future incremental replacement.
