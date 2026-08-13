---
title: Data Modeling
description: Mapping business concepts and relationships in a conceptual data model
category:
- Database
- Architecture
problems:
- poor-domain-model
- database-schema-design-problems
- complex-domain-model
- data-migration-complexities
- legacy-business-logic-extraction-difficulty
- data-structure-cache-inefficiency
- incorrect-index-type
- inefficient-database-indexing
- queries-that-prevent-index-usage
- schema-evolution-paralysis
- unused-indexes
- entity-attribute-value-overuse
- master-data-ownership-gaps
layout: solution
related_solutions:
- slug: domain-modeling
  similarity: 0.8
- slug: business-process-modeling
  similarity: 0.75
- slug: canonical-data-model
  similarity: 0.7
- slug: data-strategy
  similarity: 0.65
- slug: story-mapping
  similarity: 0.65
- slug: domain-patterns
  similarity: 0.65
---

## How to Apply ◆

- Create a conceptual data model that captures business entities and their relationships independently of the legacy system's physical schema.
- Compare the conceptual model against the legacy database schema to identify mismatches, missing concepts, and unnecessary complexity.
- Use entity-relationship diagrams to document the legacy data model and communicate it to developers and business stakeholders.
- Model data in terms of the business domain rather than technical convenience, guiding schema improvements during modernization.
- Identify data integrity constraints that exist in application code but are missing from the database schema, and document them in the data model.
- Use the data model as a blueprint for data migration planning when replacing or restructuring legacy databases.

## Tradeoffs ⇄

**Benefits:**
- Creates a shared understanding of the business data landscape across technical and business teams.
- Identifies schema design problems and opportunities for normalization or restructuring.
- Provides a foundation for data migration and system replacement planning.
- Reveals business rules embedded in database constraints or stored procedures.

**Costs:**
- Creating accurate data models for legacy systems with undocumented schemas is time-intensive.
- Data models can become outdated if not maintained alongside schema changes.
- May reveal uncomfortable truths about the gap between the ideal model and reality.
- Over-detailed models can be as hard to understand as the schemas they describe.

## How It Could Be

A legacy inventory management system has a database with over 400 tables, many with cryptic names and undocumented relationships. Before attempting a migration to a modern platform, the team creates a conceptual data model by reverse-engineering the schema and interviewing warehouse staff. They discover that thirty tables represent different versions of the same concept accumulated over years of ad-hoc extensions, and that critical business rules (such as minimum stock thresholds) are enforced only in application code, not in database constraints. The data model becomes the migration blueprint, guiding which tables to consolidate, which relationships to formalize, and which business rules to extract into the new system's domain layer.
