---
title: Data Formats
description: Use standardized and widely adopted data formats for data exchange
category:
- Architecture
problems:
- integration-difficulties
- cross-system-data-synchronization-problems
- technology-stack-fragmentation
- poor-interfaces-between-applications
- vendor-lock-in
- endianness-conversion-overhead
- alignment-and-padding-issues
layout: solution
related_solutions:
- slug: standardized-data-formats
  similarity: 0.95
- slug: data-format-conversion
  similarity: 0.85
- slug: backward-compatible-data-formats
  similarity: 0.8
- slug: data-ecosystems
  similarity: 0.75
- slug: data-strategy
  similarity: 0.75
- slug: canonical-data-model
  similarity: 0.75
---

## Description

This solution replaces proprietary, custom, or undocumented data formats used for exchange between systems with widely adopted, well-specified standards — JSON for APIs, CSV for batch transfers, Parquet for analytical workloads — chosen according to the exchange use case and accompanied by a published schema in a standard schema language. The underlying mechanism is straightforward: a standard format comes with broad tooling, library, and documentation support across languages and platforms, so any new system that needs to participate in data exchange can do so using off-the-shelf libraries instead of a bespoke parser. This is disproportionately valuable in legacy contexts because custom formats defined decades ago are frequently understood by only one remaining person, if anyone, turning every new integration into a multi-week reverse-engineering exercise instead of a routine task. Migrating away from a proprietary format is rarely a single cutover; it typically proceeds by having the legacy system support both the old and the new format for a transition period, using format validation at the boundary to catch malformed data early. The payoff is not only faster integration but reduced vendor lock-in, since a system built around open, standard formats is not tied to whatever tooling or expertise happens to still understand its original proprietary encoding.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Replace proprietary or custom data formats with widely adopted standards (JSON, XML, CSV, Parquet) for data exchange
- Choose formats based on use case: JSON for APIs, CSV for batch exports, Parquet for analytical workloads
- Define and publish schemas for all exchange formats using standard schema languages (JSON Schema, XSD)
- Migrate legacy systems gradually by supporting both proprietary and standard formats during transition
- Use format validation at system boundaries to reject malformed data early
- Document format choices and their rationale in architecture decision records

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Broad tooling and library support across languages and platforms reduces integration effort
- Lowers the barrier for new systems to participate in data exchange
- Reduces vendor lock-in by avoiding proprietary formats

**Costs and Risks:**
- Standard formats may not efficiently represent domain-specific data structures
- Migrating from proprietary formats requires conversion effort and backward compatibility handling
- Generic formats like JSON lack built-in schema enforcement, requiring additional tooling
- Some legacy systems may not have libraries available for modern standard formats

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A government agency exchanged citizen records between departments using a custom binary format defined 15 years ago. Only the original developer understood the format specification, and integrating new departments required weeks of custom parser development. By migrating to JSON with a published JSON Schema, new department integrations dropped from weeks to days, and three off-the-shelf analytics tools could consume the data without custom code.
