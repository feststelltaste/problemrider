---
title: Shared Database
description: A situation where multiple services or components share a single database.
category:
- Architecture
- Data
related_problems:
- slug: shared-dependencies
  similarity: 0.75
- slug: deployment-coupling
  similarity: 0.5
layout: problem
---

## Description
A shared database is a situation where multiple services or components share a single database. This is a common problem in monolithic architectures, where all the components are tightly coupled and deployed as a single unit. A shared database can lead to a number of problems, including deployment coupling, scaling inefficiencies, and tight coupling issues.

## Indicators ⟡
- Multiple services or components are reading from and writing to the same database.
- It is not possible to change the database schema without affecting multiple services or components.
- It is not possible to scale the database for one service or component without affecting the others.
- The database is a single point of failure for the entire system.

## Symptoms ▲

- [Deployment Coupling](deployment-coupling.md)
<br/>  Services sharing a database must coordinate deployments to avoid breaking shared schema dependencies, creating deployment coupling.
- [Scaling Inefficiencies](scaling-inefficiencies.md)
<br/>  A shared database cannot be scaled independently for different services, forcing the entire database to be scaled for peak demand of any single consumer.
- [Schema Evolution Paralysis](schema-evolution-paralysis.md)
<br/>  Schema changes become risky and difficult when multiple services depend on the same tables, leading to schema evolution paralysis.
- [Team Coordination Issues](team-coordination-issues.md)
<br/>  Teams owning different services must coordinate database changes, creating communication overhead and cross-team dependencies.
- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  Services become tightly coupled through their shared data model, making it impossible to change one without considering all others.

## Causes ▼
- [Monolithic Architecture Constraints](monolithic-architecture-constraints.md)
<br/>  Monolithic systems naturally use a single shared database, and this pattern persists even when services are extracted from the monolith.
- [Convenience-Driven Development](convenience-driven-development.md)
<br/>  Sharing a database is the easiest path for new services to access existing data, leading teams to choose convenience over proper decoupling.
- [Short-Term Focus](short-term-focus.md)
<br/>  Management prioritizes quick feature delivery over database decoupling, perpetuating shared database patterns that create long-term problems.

## Detection Methods ○
- **Architectural Diagrams:** Create a diagram of the system architecture to identify which services or components are sharing a single database.
- **Database Schema Analysis:** Analyze the database schema to identify which tables are being used by multiple services or components.
- **Developer Surveys:** Ask developers if they feel like they are able to change the database schema without affecting other services or components.

## Examples
A company has a large, monolithic e-commerce application. The application is composed of a number of different services, including a product catalog, a shopping cart, and a payment gateway. All of the services share a single database. When the development team wants to make a change to the database schema for the product catalog, they have to be careful not to break the shopping cart or the payment gateway. This makes it difficult to make changes to the database, and it often leads to problems.
