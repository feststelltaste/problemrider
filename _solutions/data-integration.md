---
title: Data Integration
description: Merging data from various sources and providing it uniformly
category:
- Database
- Architecture
problems:
- cross-system-data-synchronization-problems
- information-fragmentation
- shared-database
- data-migration-complexities
- poor-interfaces-between-applications
- integration-difficulties
layout: solution
related_solutions:
- slug: data-ecosystems
  similarity: 0.8
- slug: data-strategy
  similarity: 0.75
- slug: canonical-data-model
  similarity: 0.75
- slug: business-event-processing
  similarity: 0.7
- slug: data-integrity
  similarity: 0.7
- slug: data-replication
  similarity: 0.7
---

## Description

Data integration merges data scattered across multiple legacy systems into a coherent, uniformly accessible view, typically through a dedicated integration layer — ETL pipelines, data virtualization, or event-based synchronization — rather than through direct point-to-point connections between each pair of systems. The approach starts by mapping how the same business entities are represented differently across systems, then defines a canonical model for each shared entity that serves as the contract the integration layer translates every source into, cleansing and validating data at that boundary rather than propagating each source's quality problems downstream. This is especially relevant for legacy landscapes assembled over years through separate departmental systems, acquisitions, or uncoordinated expansion, where the same customer, patient, or product exists in several systems with no shared identity and where users are left to manually cross-reference multiple screens to reconstruct a single coherent picture. Where a legacy database cannot be modified to emit events directly, change data capture lets the integration layer observe changes at the database level and propagate them without touching the source application at all. Because the integration layer becomes a piece of critical infrastructure that every consuming system now depends on, its own reliability, monitoring, and latency characteristics need as much attention as the data quality problems it was built to solve.

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
