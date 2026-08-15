---
title: Standardized Protocols
description: Select transport and messaging protocols with broad ecosystem support
category:
- Architecture
- Dependencies
problems:
- poor-interfaces-between-applications
- technology-lock-in
- vendor-lock-in
- integration-difficulties
- obsolete-technologies
- microservice-communication-overhead
layout: solution
related_solutions:
- slug: standardized-interfaces
  similarity: 0.75
- slug: protocol-abstraction
  similarity: 0.75
- slug: standardized-data-formats
  similarity: 0.75
- slug: platform-independent-data-storage
  similarity: 0.7
- slug: schema-registry
  similarity: 0.7
- slug: service-mesh
  similarity: 0.7
---

## Description

Standardized protocols means selecting transport and messaging protocols with broad, multi-vendor ecosystem support — HTTP/2, AMQP, MQTT, gRPC — in place of proprietary protocols that only a single vendor's tooling or middleware can speak. Legacy systems, especially in industrial or telecommunications contexts, often communicate using a protocol that was proprietary from the start or that became a de facto lock-in mechanism over time, meaning the organization's ability to integrate, monitor, or even keep the system running depends entirely on one vendor's continued goodwill, licensing terms, and support lifecycle. This dependency becomes acute the moment that vendor changes its pricing, discontinues the middleware, or simply becomes harder to find developers for, at which point the organization discovers it has no real alternative. Migrating to a standardized protocol — typically via an interim protocol bridge or adapter placed in front of the legacy system so the transition can happen incrementally rather than as a single risky cutover — restores the ability to choose from a broad, competitive ecosystem of tools, libraries, and available engineering talent. The corresponding costs are the development and testing effort of the migration itself, the operational overhead of running a bridge during the transition period, and the possibility that a standardized protocol lacks some specialized feature the proprietary one provided, which must be evaluated before committing to the change.

## How to Apply ◆

- Inventory all communication protocols currently used across the legacy landscape and identify proprietary or obsolete ones.
- Select widely supported protocols (HTTP/2, AMQP, MQTT, gRPC) based on the communication patterns required (request-response, event streaming, pub-sub).
- Introduce protocol bridges or adapters to allow legacy systems using proprietary protocols to communicate with systems using standard protocols during a transition period.
- Migrate legacy integrations from proprietary protocols to standardized ones incrementally, starting with the highest-traffic or most-problematic connections.
- Ensure chosen protocols are supported by the target platforms and languages used across the organization.

## Tradeoffs ⇄

**Benefits:**
- Broad ecosystem support means readily available libraries, tools, and developer knowledge.
- Reduces vendor lock-in by avoiding proprietary communication mechanisms.
- Simplifies integration with external partners and third-party services.
- Makes it easier to find developers who understand the technology.

**Costs:**
- Migrating from proprietary protocols requires development effort and careful testing.
- Standardized protocols may lack specialized features that proprietary protocols offered.
- Running protocol bridges during transition adds operational complexity.
- Some legacy systems may not support modern protocols without significant modification.

## How It Could Be

A manufacturing company's legacy SCADA systems communicate using a proprietary binary protocol that only one vendor's middleware can handle. When the vendor raises licensing fees significantly, the team decides to migrate to MQTT for device-to-server communication and AMQP for inter-service messaging. They deploy protocol adapters at the boundary of legacy systems that cannot be immediately modified. New services are built using the standard protocols from the start. Within a year, the vendor dependency is eliminated for most communication paths, and the team can choose from multiple open-source tools for monitoring and managing their messaging infrastructure.
