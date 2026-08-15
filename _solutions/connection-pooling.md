---
title: Connection Pooling
description: Reusing pre-established connections instead of creating new ones per
  request
category:
- Performance
- Database
problems:
- database-connection-leaks
- misconfigured-connection-pools
- high-connection-count
- slow-application-performance
- incorrect-max-connection-pool-size
- high-database-resource-utilization
- unreleased-resources
layout: solution
related_solutions:
- slug: distributed-caching
  similarity: 0.8
- slug: resource-pooling
  similarity: 0.8
- slug: lazy-loading
  similarity: 0.8
- slug: reactive-programming
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.75
- slug: parallelization
  similarity: 0.75
---

## Description

Connection pooling maintains a set of pre-established, reusable connections to a resource — most commonly a database, but equally applicable to HTTP clients, LDAP servers, or message brokers — so that requests borrow a ready connection from the pool instead of paying the cost of establishing and tearing down a new one every time. Establishing a connection typically involves a TCP handshake, authentication, and often TLS negotiation, all of which are fixed costs that scale with request volume rather than with actual work done; under load this overhead alone can push a database to its connection limit long before it runs out of real capacity. Legacy applications frequently create a connection per request out of simplicity, a pattern that was invisible at low traffic and only becomes a bottleneck as usage grows, often compounded by connection leaks where code acquires a connection but never returns it. Pooling caps the number of concurrent connections to a size the backing resource can actually sustain, which both improves response times by removing setup latency from the request path and protects the resource from being overwhelmed by unbounded connection growth. Getting the pool size, timeout, and validation settings right is essential, since an undersized or misconfigured pool merely relocates the bottleneck from connection creation to connection queuing.

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
