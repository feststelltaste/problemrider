---
title: Service Mesh
description: Managing traffic at infrastructure level with transparent protocol translation, mTLS, and routing
category:
- Architecture
- Operations
quality_tactics_url: https://qualitytactics.de/en/compatibility/service-mesh
problems:
- microservice-communication-overhead
- service-discovery-failures
- service-timeouts
- network-latency
- insecure-data-transmission
- monitoring-gaps
- cascade-failures
layout: solution
---

## How to Apply ◆

- Deploy a service mesh (e.g., Istio, Linkerd) as a sidecar proxy layer alongside existing legacy services to gain traffic management without modifying application code.
- Enable mTLS between services to secure communication channels that legacy systems may have left unencrypted.
- Use the mesh's traffic routing capabilities to implement canary deployments and gradual migration from legacy to modernized services.
- Configure retry policies, circuit breakers, and timeouts at the infrastructure level to improve resilience of legacy service interactions.
- Leverage built-in observability (distributed tracing, metrics) to gain visibility into legacy service communication patterns.
- Use protocol translation features to bridge legacy protocols with modern ones without rewriting service code.

## Tradeoffs ⇄

**Benefits:**
- Adds security, observability, and resilience to legacy services without requiring code changes.
- Enables gradual traffic shifting during migration from legacy to modern services.
- Provides consistent traffic policies across heterogeneous legacy and modern components.
- Centralizes cross-cutting concerns like retries, timeouts, and authentication.

**Costs:**
- Introduces significant infrastructure complexity and operational overhead.
- Sidecar proxies add latency and resource consumption to every service call.
- Debugging becomes harder because requests pass through additional proxy layers.
- Requires container orchestration (typically Kubernetes), which legacy environments may not have.
- Steep learning curve for operations teams unfamiliar with mesh concepts.

## Examples

An e-commerce platform runs a mix of legacy Java services and newer microservices. Inter-service communication is unreliable, with frequent timeouts and no encryption. The team deploys Linkerd as a service mesh, starting with the most critical communication paths. The mesh automatically provides mTLS, retries with backoff, and detailed latency metrics. During a subsequent migration phase, they use traffic splitting to route 10% of requests to a rewritten service while 90% still go to the legacy version, allowing safe validation before full cutover. The observability data from the mesh also reveals previously unknown dependency chains between legacy services.
