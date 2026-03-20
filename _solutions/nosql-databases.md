---
title: NoSQL Databases
description: Storing data in flexible, schema-less formats
category:
- Database
- Architecture
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/nosql-databases
problems:
- database-schema-design-problems
- scaling-inefficiencies
- schema-evolution-paralysis
- slow-database-queries
- unbounded-data-growth
- high-database-resource-utilization
- data-migration-complexities
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify use cases where the relational model creates friction: highly variable schemas, document-oriented data, or extreme read/write volumes
- Choose the appropriate NoSQL category (document, key-value, column-family, graph) based on the specific access patterns
- Start by offloading specific workloads (e.g., session storage, event logs, product catalogs) rather than replacing the entire relational database
- Implement a data access layer that abstracts the storage backend so the rest of the application is not tightly coupled to the NoSQL technology
- Plan for eventual consistency if moving from a strongly consistent relational system, ensuring the business logic can tolerate it
- Migrate data incrementally, running both stores in parallel during the transition period with reconciliation checks

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables horizontal scaling for workloads that outgrow single-server relational databases
- Eliminates painful schema migrations for data with evolving or variable structures
- Can dramatically improve read and write performance for specific access patterns
- Reduces impedance mismatch between application objects and storage format

**Costs and Risks:**
- Loss of ACID transactions and strong consistency requires careful application-level handling
- Teams experienced only with relational databases face a significant learning curve
- Running multiple database technologies increases operational complexity
- Lack of schema enforcement can lead to data quality issues over time
- Query capabilities are often more limited than SQL, pushing complexity into application code

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An e-commerce company's legacy relational database struggled with a product catalog where each product category had vastly different attributes, resulting in hundreds of sparse columns and a proliferation of entity-attribute-value tables. The team migrated the catalog to MongoDB, storing each product as a document with category-specific fields. This eliminated the complex JOIN queries that had been causing slow page loads, reduced the catalog query response time from seconds to milliseconds, and made it trivial for merchandising teams to add new product attributes without filing database change requests.
