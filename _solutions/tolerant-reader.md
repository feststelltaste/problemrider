---
title: Tolerant Reader Pattern
description: Ignoring unknown fields and tolerating structural additions on the consumer side
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/compatibility/tolerant-reader
problems:
- breaking-changes
- api-versioning-conflicts
- integration-difficulties
- brittle-codebase
- ripple-effect-of-changes
- tight-coupling-issues
layout: solution
---

## How to Apply ◆

- Configure deserializers in consumer services to ignore unknown fields rather than failing on unexpected properties (e.g., `DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES = false` in Jackson).
- Design consumers to extract only the fields they need, avoiding strict schema binding to the full message structure.
- Write consumer-side tests that verify behavior remains correct when extra fields are added to payloads.
- Apply the pattern when wrapping legacy APIs: build tolerant adapters that gracefully handle variations in legacy system responses.
- Document which fields a consumer actually depends on, making the implicit contract explicit.
- Use the pattern alongside schema evolution strategies to allow producers to add new fields without coordinating with every consumer.

## Tradeoffs ⇄

**Benefits:**
- Producers can evolve their schemas by adding fields without breaking existing consumers.
- Reduces the coordination overhead required for changes across multiple teams.
- Increases system resilience by preventing failures from minor structural changes.
- Particularly valuable in legacy systems where multiple consumers depend on the same data source.

**Costs:**
- Consumers may silently miss important new fields that they should be processing.
- Can mask real incompatibilities if consumers are too tolerant of structural changes.
- Makes it harder to detect when a consumer's understanding of a contract has drifted from reality.
- Requires discipline to ensure consumers validate the fields they do use.

## How It Could Be

A legacy ERP system publishes order events consumed by five downstream services. Each time the ERP team adds a field to the order payload, at least one consumer breaks because its strict deserialization rejects the unknown property. After adopting the tolerant reader pattern, consumers are configured to ignore unrecognized fields and extract only the data they need. The ERP team can now enrich order events with new attributes (shipping metadata, compliance flags) without filing cross-team change requests. Consumers that need the new data opt in by updating their extraction logic on their own schedule, while those that do not need it continue operating without any changes.
