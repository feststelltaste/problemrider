---
title: Data Integration
description: Merging data from various sources and providing it uniformly
category:
- Database
- Architecture
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/data-integration
problems:
- cross-system-data-synchronization-problems
- information-fragmentation
- shared-database
- data-migration-complexities
- poor-interfaces-between-applications
- integration-difficulties
layout: solution
---

## How to Apply ◆

- Map data entities across legacy systems to identify overlaps, conflicts, and semantic differences in how the same concepts are represented.
- Implement an integration layer (ETL pipelines, data virtualization, or event-based synchronization) rather than point-to-point connections between legacy systems.
- Define canonical data models for shared entities that serve as the integration contract between systems.
- Handle data quality issues at the integration boundary: validate, cleanse, and transform data as it flows between systems.
- Use change data capture (CDC) for near-real-time integration with legacy databases that cannot be modified to emit events.
- Monitor data integration pipelines with alerting for synchronization failures, data quality drops, and latency increases.

## Tradeoffs ⇄

**Benefits:**
- Provides a unified view of data scattered across legacy systems, enabling reporting and analytics.
- Reduces data inconsistencies caused by manual re-entry across systems.
- Decouples systems by routing data through an integration layer rather than direct database sharing.
- Enables incremental system replacement by allowing new systems to consume integrated data feeds.

**Costs:**
- Building and maintaining integration pipelines is a significant ongoing investment.
- Data mapping across legacy systems with inconsistent schemas is complex and error-prone.
- Integration introduces latency; real-time consistency across systems may not be achievable.
- Integration layer becomes critical infrastructure; its failure impacts all connected systems.

## How It Could Be

A hospital runs separate legacy systems for patient registration, billing, lab results, and pharmacy. Clinicians must log into multiple systems and manually cross-reference patient information, leading to delays and occasional errors. The IT team implements a data integration platform using Apache NiFi, creating pipelines that synchronize patient demographics across systems and provide a unified patient record view. Change data capture on the registration system's database feeds updates to downstream systems in near-real-time. The integration layer normalizes data formats and resolves conflicts (such as different date formats and name representations) before delivering data to consumers. Clinicians now see a consolidated patient view, and the integration layer provides the foundation for eventually replacing individual legacy systems.
