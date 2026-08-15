---
title: Service Mesh
description: Managing traffic at infrastructure level with transparent protocol translation,
  mTLS, and routing
category:
- Architecture
- Operations
problems:
- microservice-communication-overhead
- service-discovery-failures
- service-timeouts
- network-latency
- insecure-data-transmission
- monitoring-gaps
- cascade-failures
layout: solution
related_solutions:
- slug: containerization
  similarity: 0.75
- slug: strangler-fig-pattern
  similarity: 0.7
- slug: microservices-architecture
  similarity: 0.7
- slug: microservices
  similarity: 0.7
- slug: api-gateway
  similarity: 0.7
- slug: chaos-engineering
  similarity: 0.7
---

## Description

A service mesh is an infrastructure layer, typically implemented as a set of sidecar proxies deployed alongside each service instance, that intercepts and manages all network traffic between services without requiring changes to application code. It centralizes cross-cutting communication concerns — mutual TLS encryption, retries, timeouts, circuit breaking, load balancing, and distributed tracing — that would otherwise need to be implemented redundantly, and often inconsistently, inside every service. This externalization is particularly valuable for legacy systems, where inter-service communication frequently predates modern security and resilience practices: connections may be unencrypted, timeout and retry behavior may be hard-coded or entirely absent, and there is often no visibility into how legacy components actually talk to one another until a mesh's tracing surfaces the real dependency graph. Because the mesh operates at the network layer rather than inside application code, it can be introduced incrementally around existing legacy services, adding protocol translation, traffic shaping, and security controls as a wrapper rather than as a rewrite. This same traffic-shaping capability makes the mesh a practical mechanism for gradual migration, since a percentage of traffic can be routed to a modernized replacement service while the rest continues to flow to the legacy implementation, allowing behavior to be validated under real load before a full cutover.

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

## How It Could Be

An e-commerce platform runs a mix of legacy Java services and newer microservices. Inter-service communication is unreliable, with frequent timeouts and no encryption. The team deploys Linkerd as a service mesh, starting with the most critical communication paths. The mesh automatically provides mTLS, retries with backoff, and detailed latency metrics. During a subsequent migration phase, they use traffic splitting to route 10% of requests to a rewritten service while 90% still go to the legacy version, allowing safe validation before full cutover. The observability data from the mesh also reveals previously unknown dependency chains between legacy services.
