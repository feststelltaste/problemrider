---
title: Connection Pooling
description: Reusing pre-established connections instead of creating new ones per request
category:
- Performance
- Database
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/connection-pooling
problems:
- database-connection-leaks
- misconfigured-connection-pools
- high-connection-count
- slow-application-performance
- incorrect-max-connection-pool-size
- high-database-resource-utilization
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Replace direct connection creation with a connection pool library appropriate for the technology stack (HikariCP, pgBouncer, c3p0)
- Size the pool based on actual concurrent usage patterns, not arbitrary large numbers
- Configure appropriate connection validation and eviction policies to handle stale or broken connections
- Set connection timeouts and maximum wait times so the application fails fast rather than hanging
- Monitor pool metrics: active connections, idle connections, wait times, and connection creation rates
- Audit legacy code for connection leaks where connections are acquired but not properly returned to the pool
- Apply connection pooling to all external resources: databases, HTTP clients, LDAP connections, message brokers

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates the overhead of establishing new connections for each request (TCP handshake, authentication, SSL negotiation)
- Provides predictable resource consumption by capping the maximum number of connections
- Improves response times by having pre-established, ready-to-use connections available
- Reduces load on the database server by limiting concurrent connections

**Costs and Risks:**
- Incorrectly sized pools can cause connection starvation (too small) or resource waste (too large)
- Stale connections in the pool can cause intermittent failures if validation is not configured
- Connection pools add configuration complexity that must be tuned for the specific workload
- Pool exhaustion under load can cause cascading failures if not handled with proper timeouts

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy Java web application created a new database connection for every HTTP request and closed it at the end of the request handler. Under load, the database server hit its maximum connection limit, causing new requests to fail with connection refused errors. The team introduced HikariCP with a pool of 20 connections, matching the database's recommended maximum for the application. Connection establishment overhead disappeared from the request path, average response times improved by 15%, and the database server's CPU usage dropped because it no longer spent cycles managing thousands of short-lived connections per minute.
