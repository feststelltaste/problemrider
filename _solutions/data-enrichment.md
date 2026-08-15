---
title: Data Enrichment
description: Supplementing data with additional information from external sources
category:
- Database
- Dependencies
problems:
- poor-domain-model
- feature-gaps
- data-migration-complexities
- cross-system-data-synchronization-problems
- silent-data-corruption
layout: solution
related_solutions:
- slug: data-quality-checks
  similarity: 0.7
- slug: data-integration
  similarity: 0.65
- slug: data-integrity
  similarity: 0.65
- slug: data-deduplication
  similarity: 0.6
- slug: continuous-data-verification
  similarity: 0.6
- slug: data-strategy
  similarity: 0.6
---

## Description

Data enrichment supplements existing records with additional attributes drawn from external sources — reference databases, commercial data providers, or other internal systems — rather than requiring that missing or outdated information be captured through manual re-entry. An enrichment pipeline typically runs on ingestion or on a schedule, matches legacy records against the external source using available identifiers, and writes the resulting fields into the system either as additions to the original record or, more safely, into a separate store that preserves which values came from the legacy system and which were added later. This technique is particularly relevant to legacy systems because their data was frequently captured under older, more limited business processes and has degraded or fallen out of date over years of operation, creating gaps — missing classifications, stale contact details, absent geolocation — that block new features or analytics the organization now wants to build on top of old data. Because enrichment introduces a dependency on an external source's availability, quality, and update cadence, pipelines need validation against business rules and a fallback strategy for when that source cannot be reached. Preserving lineage between original and enriched values is essential, since it keeps the enrichment auditable and reversible if the external source later proves to be wrong.

## How to Apply ◆

- Identify gaps in legacy data that reduce system effectiveness (e.g., missing geolocation, outdated contact information, incomplete classification).
- Integrate external data sources (APIs, reference databases, third-party services) to supplement legacy data.
- Build enrichment pipelines that run on ingestion or on a schedule, adding derived or supplemental fields to legacy records.
- Validate enriched data against business rules to prevent introducing errors into the legacy system.
- Store enrichment results separately from original data to preserve data lineage and allow rollback.
- Monitor enrichment quality over time and establish fallback strategies for when external sources are unavailable.

## Tradeoffs ⇄

**Benefits:**
- Improves the quality and completeness of legacy data without requiring manual data entry.
- Enables new features and analytics capabilities that the legacy data alone cannot support.
- Can correct or supplement data that has degraded over years of legacy system operation.

**Costs:**
- Introduces dependencies on external data sources with their own availability and quality concerns.
- Enrichment processes add complexity to the data pipeline and require ongoing maintenance.
- Privacy and compliance considerations may limit which external data can be integrated.
- Incorrect enrichment can introduce errors that are difficult to distinguish from original data.

## How It Could Be

A legacy customer database contains millions of records accumulated over twenty years, many with incomplete addresses, missing industry classifications, and outdated contact information. The team builds an enrichment pipeline that matches customer records against a commercial business data provider, filling in missing fields and flagging records where stored information conflicts with external sources. The enrichment results are stored in a separate table linked to the original records, preserving the ability to audit what came from the legacy system versus what was enriched. Sales teams immediately benefit from improved targeting, and the data quality improvements enable a customer segmentation feature that was previously impossible due to incomplete data.
