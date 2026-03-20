---
title: Data Enrichment
description: Supplementing data with additional information from external sources
category:
- Database
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/data-enrichment
problems:
- poor-domain-model
- feature-gaps
- data-migration-complexities
- cross-system-data-synchronization-problems
- silent-data-corruption
layout: solution
---

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
