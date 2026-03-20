---
title: Standardized Protocols
description: Select transport and messaging protocols with broad ecosystem support
category:
- Architecture
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/compatibility/standardized-protocols
problems:
- poor-interfaces-between-applications
- technology-lock-in
- vendor-lock-in
- integration-difficulties
- obsolete-technologies
- microservice-communication-overhead
layout: solution
---

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
