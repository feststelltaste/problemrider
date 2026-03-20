---
title: Data Formats
description: Use standardized and widely adopted data formats for data exchange
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/compatibility/data-formats
problems:
- integration-difficulties
- cross-system-data-synchronization-problems
- technology-stack-fragmentation
- poor-interfaces-between-applications
- vendor-lock-in
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A government agency exchanged citizen records between departments using a custom binary format defined 15 years ago. Only the original developer understood the format specification, and integrating new departments required weeks of custom parser development. By migrating to JSON with a published JSON Schema, new department integrations dropped from weeks to days, and three off-the-shelf analytics tools could consume the data without custom code.
