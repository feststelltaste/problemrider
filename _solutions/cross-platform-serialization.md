---
title: Cross-Platform Serialization
description: Use data serializers that are compatible across different systems
category:
- Architecture
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/compatibility/cross-platform-serialization
problems:
- cross-system-data-synchronization-problems
- integration-difficulties
- serialization-deserialization-bottlenecks
- technology-stack-fragmentation
- poor-interfaces-between-applications
- breaking-changes
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Replace language-specific serialization (Java Serializable, .NET BinaryFormatter, Python pickle) with platform-neutral formats
- Choose a serialization format appropriate for your use case: JSON for human-readable APIs, Protocol Buffers or Avro for high-throughput internal communication
- Define schemas for serialized data and version them explicitly
- Test serialization and deserialization across all platforms that exchange data
- Implement tolerant readers that handle unknown fields gracefully during schema evolution
- Migrate incrementally by supporting both old and new serialization formats during transition

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables data exchange between systems written in different languages and frameworks
- Reduces risk of deserialization vulnerabilities associated with language-native serialization
- Simplifies adding new systems to the integration landscape

**Costs and Risks:**
- Platform-neutral formats may be less performant than native binary serialization
- Schema management adds complexity, especially when multiple versions coexist
- Migration from proprietary serialization formats requires careful backward compatibility handling
- Some complex object graphs may be difficult to represent in simpler cross-platform formats

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A logistics company had a Java-based order system using Java Serializable to store messages in a queue, which prevented a new Python-based analytics service from consuming those messages. The team migrated the message format to Avro with a schema registry, running both formats in parallel for four weeks. After the transition, both Java and Python services consumed the same message stream without any translation layer, and the schema registry prevented three incompatible schema changes during subsequent development.
