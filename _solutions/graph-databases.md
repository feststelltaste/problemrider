---
title: Graph Databases
description: Enable the storage and querying of connected data in the form of nodes and edges
category:
- Database
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/graph-databases
problems:
- slow-database-queries
- database-query-performance-issues
- algorithmic-complexity-problems
- database-schema-design-problems
- complex-domain-model
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify data models in the legacy system that are inherently graph-shaped: social networks, organizational hierarchies, dependency trees, recommendation engines
- Evaluate whether current performance problems stem from deep JOIN chains or recursive queries that would benefit from graph traversal
- Choose a graph database (Neo4j, Amazon Neptune, JanusGraph) appropriate for the scale and query patterns
- Model the domain as nodes (entities) and edges (relationships) with properties on both
- Migrate the graph-shaped subset of data to the graph database while keeping non-graph data in the relational store
- Implement synchronization between the relational and graph databases if both must remain consistent
- Use graph query languages (Cypher, Gremlin) to express relationship traversals that are cumbersome in SQL

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Excels at traversing relationships, which can be orders of magnitude faster than SQL JOINs for deep graph queries
- Natural modeling of connected data without complex join tables
- Query performance remains stable as data grows because traversal depends on local graph structure, not total dataset size
- Enables discovery of patterns and paths that are impractical to query in relational databases

**Costs and Risks:**
- Adds another database technology to the stack, increasing operational complexity
- Graph databases are less mature and have smaller ecosystems than relational databases
- Not suitable for all workloads: tabular data and aggregations are better handled by relational databases
- Team needs to learn new query languages and data modeling paradigms

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy access control system stored organizational hierarchies and permission inheritance in a relational database. Determining a user's effective permissions required recursive queries through multiple levels of group membership and role inheritance, which took over 10 seconds for deeply nested organizations. The team migrated the permission model to Neo4j, where a Cypher query could traverse the entire permission graph in milliseconds regardless of depth. The relational database remained the authoritative source for user and group data, with changes synchronized to Neo4j via events. Permission checks dropped from seconds to single-digit milliseconds, and the system could now support organizations with arbitrarily deep hierarchies.
