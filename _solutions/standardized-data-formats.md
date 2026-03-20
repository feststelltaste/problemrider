---
title: Standardized Data Formats
description: Use widely adopted, platform-independent data formats for data exchange
category:
- Architecture
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/portability/standardized-data-formats
problems:
- technology-lock-in
- vendor-lock-in
- poor-interfaces-between-applications
- cross-system-data-synchronization-problems
- data-migration-complexities
- serialization-deserialization-bottlenecks
- integration-difficulties
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Inventory all data exchange points in the system including APIs, file imports/exports, message queues, and inter-service communication
- Replace proprietary or custom binary formats with standardized alternatives such as JSON, XML, CSV, Protocol Buffers, or Apache Avro
- Define schemas for all data formats using standards like JSON Schema, XML Schema, or Avro schemas to enforce structure
- Introduce format validation at system boundaries to reject malformed data early
- Use content negotiation in APIs so consumers can request data in their preferred standard format
- Document all data formats and schemas, making them available to integration partners and internal teams

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables interoperability with a wide range of systems and platforms without custom parsers
- Reduces integration effort since standard formats have mature libraries in every major language
- Makes data migration between systems feasible by using universally understood formats
- Improves data longevity since standardized formats are less likely to become obsolete

**Costs and Risks:**
- Text-based formats like JSON and XML are less efficient than binary formats for large data volumes
- Migrating from proprietary formats requires careful mapping and validation to prevent data loss
- Schema evolution must be managed deliberately to maintain backward compatibility
- Some domain-specific data may not map cleanly to generic standardized formats

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A manufacturing company exchanged production data between its legacy ERP system and supplier portals using a proprietary binary format developed in-house 15 years earlier. Only two developers understood the format, and every new integration partner required weeks of custom adapter development. The team migrated to JSON with published JSON Schemas for each data exchange type. Existing integrations were updated using a format translation layer that converted between the legacy binary format and JSON. New integration partners could begin development immediately using standard tools, reducing onboarding time from weeks to days.
