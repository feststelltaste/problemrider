---
title: Microservices Architecture
description: Divide application into small, independent services
category:
- Architecture
problems:
- monolithic-architecture-constraints
- deployment-coupling
- tight-coupling-issues
- scaling-inefficiencies
- high-coupling-low-cohesion
- technology-lock-in
- slow-development-velocity
- large-risky-releases
- stagnant-architecture
layout: solution
related_solutions:
- slug: microservices
  similarity: 0.9
- slug: containerization
  similarity: 0.75
- slug: modulith
  similarity: 0.75
- slug: layered-architecture
  similarity: 0.75
- slug: hexagonal-architecture
  similarity: 0.75
- slug: service-mesh
  similarity: 0.7
---

## Description

Microservices architecture decomposes a system into a set of small, independently deployable services, each aligned to a specific business capability, communicating over the network through well-defined APIs rather than through in-process method calls or a shared database. Extraction from an existing legacy monolith is typically done incrementally through the Strangler Fig pattern, identifying bounded contexts using domain-driven design, and routing traffic to new services one at a time while the monolith continues to handle everything not yet extracted, rather than attempting a wholesale rewrite. Legacy monoliths tend to reach a point where every change, however small, requires redeploying and retesting the entire application because all of its subsystems are compiled and shipped together, which slows delivery, concentrates risk in every release, and makes it impossible for one team to scale or evolve their part of the system independently of every other team's schedule. Microservices directly address this by making the unit of deployment, scaling, and technology choice the individual service rather than the whole application, which also creates a natural mapping between service ownership and team boundaries and allows legacy technology to be replaced gradually, service by service, instead of all at once. The tradeoff legacy teams take on in return is a substantial increase in distributed systems complexity — network latency, partial failures, and cross-service data consistency — that a monolith never had to deal with, and premature or overly fine-grained decomposition of a poorly understood legacy system frequently produces a "distributed monolith" that is harder to operate than the system it replaced.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify bounded contexts in the legacy monolith using domain-driven design techniques
- Start by extracting the least coupled, most independently deployable module as the first microservice
- Use the Strangler Fig pattern to incrementally route traffic from the monolith to new services
- Define clear API contracts between services using REST, gRPC, or messaging before splitting codebases
- Introduce an API gateway or service mesh to manage routing, authentication, and observability
- Set up independent CI/CD pipelines for each service to enable autonomous team deployments
- Implement distributed tracing and centralized logging from the start to maintain operational visibility
- Plan a data decomposition strategy so each service owns its data rather than sharing a single database

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables independent deployment and scaling of individual services
- Allows different services to use different technology stacks suited to their problem domain
- Reduces the blast radius of failures and changes to a single service boundary
- Facilitates team autonomy by aligning service ownership with team boundaries
- Breaks vendor lock-in by allowing gradual platform migration service by service

**Costs and Risks:**
- Introduces distributed systems complexity including network latency, partial failures, and data consistency challenges
- Requires significant investment in infrastructure, monitoring, and operational tooling
- Premature decomposition can create a distributed monolith that is harder to manage than the original
- Cross-service refactoring and schema changes become more difficult to coordinate
- Teams need strong DevOps capabilities to manage independent service lifecycles

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An e-commerce company had a monolithic application where deploying a change to the recommendation engine required redeploying the entire system, including checkout and inventory management. The team used domain-driven design workshops to identify five bounded contexts and began extracting the recommendation service first, since it had the fewest database dependencies. Over eighteen months they extracted four services using the Strangler Fig pattern, routing traffic gradually from the monolith. Each service gained its own deployment pipeline and could be scaled independently. The recommendation service was migrated to Python for its ML libraries while the checkout service stayed on Java, demonstrating the technology flexibility that microservices provide.
