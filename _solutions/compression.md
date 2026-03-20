---
title: Compression
description: Reduce storage space with or without loss
category:
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/compression
problems:
- slow-application-performance
- network-latency
- excessive-disk-io
- unbounded-data-growth
- high-client-side-resource-consumption
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Enable HTTP compression (gzip, Brotli) on web servers and API gateways for text-based responses
- Compress data at rest in databases and file storage for archival and infrequently accessed data
- Use protocol-level compression for inter-service communication (gRPC, compressed message queue payloads)
- Choose compression algorithms appropriate to the data type and access pattern: fast algorithms for real-time, high-ratio algorithms for archival
- Implement compression for log files and audit trails that grow unboundedly
- Test compression ratios and CPU overhead with representative production data before deploying

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces network bandwidth consumption, improving transfer times especially over slow connections
- Decreases storage costs for data at rest
- Improves cache efficiency by fitting more data into limited cache space
- Can significantly improve page load times for web applications

**Costs and Risks:**
- Compression and decompression consume CPU cycles, which may be a bottleneck for CPU-bound systems
- Lossy compression (for images, audio) permanently reduces data quality
- Compressed data is harder to inspect and debug without decompression tools
- Some data types (already compressed images, encrypted data) do not compress well and waste CPU

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy document management system stored XML-based documents uncompressed, consuming over 8 TB of storage that was growing by 500 GB per quarter. API responses for document retrieval were slow due to the large payload sizes. The team enabled gzip compression on the API gateway, reducing response sizes by approximately 85% for XML payloads. For storage, they implemented transparent compression at the database level for documents older than 90 days. These changes freed 5 TB of storage immediately and reduced document retrieval times from 3 seconds to under 500 milliseconds for typical documents.
