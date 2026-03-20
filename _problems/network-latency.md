---
title: Network Latency
description: Delays in data transmission across the network significantly increase
  response times and impact application performance.
category:
- Performance
related_problems:
- slug: high-api-latency
  similarity: 0.8
- slug: slow-application-performance
  similarity: 0.65
- slug: lazy-loading
  similarity: 0.65
- slug: external-service-delays
  similarity: 0.6
- slug: excessive-disk-io
  similarity: 0.6
- slug: service-timeouts
  similarity: 0.6
solutions:
- caching-strategy
- serialization-optimization
layout: problem
---

## Description
Network latency is the time it takes for data to travel from one point to another on a network. While some latency is unavoidable, high network latency can have a significant impact on application performance, especially in distributed systems where services communicate over the network. This can manifest as slow response times, timeouts, and a generally sluggish user experience. Understanding and mitigating the impact of network latency is a key consideration in the design of distributed systems.

## Indicators ⟡
- Your application is slow, but your servers are not under heavy load.
- You see a high number of timeout errors in your logs.
- Your application's performance is inconsistent.
- You are getting complaints from users about slow performance.

## Symptoms ▲

- [Slow Application Performance](slow-application-performance.md)
<br/>  Network delays add directly to request processing time, degrading overall application responsiveness.
- [Service Timeouts](service-timeouts.md)
<br/>  High network latency causes inter-service communication to exceed timeout thresholds, triggering timeout errors.
- [Negative User Feedback](negative-user-feedback.md)
<br/>  Users experience sluggish interactions due to network delays and complain about poor performance.
- [High API Latency](high-api-latency.md)
<br/>  Network transmission delays directly increase API response times measured at the client.
- [External Service Delays](external-service-delays.md)
<br/>  Network transmission delays directly contribute to slow responses from external services that the application depends on.
## Causes ▼

- [Microservice Communication Overhead](microservice-communication-overhead.md)
<br/>  APIs that require many round-trips to complete operations amplify the impact of network latency on overall performance.
## Detection Methods ○

- **Ping/Traceroute:** Use `ping` to measure round-trip time to a host and `traceroute` (or `tracert` on Windows) to identify the path and latency at each hop.
- **Network Monitoring Tools:** Use tools like Wireshark, tcpdump, or network performance monitoring solutions to analyze network traffic and identify bottlenecks.
- **Distributed Tracing:** Trace requests across services to see how much time is spent in network communication versus actual processing.
- **Real User Monitoring (RUM):** RUM tools can measure network latency experienced by actual users from different locations.
- **Cloud Provider Metrics:** If using cloud services, monitor network I/O and latency metrics provided by the cloud provider.

## Examples
A company has its main application servers in North America, but a significant portion of its user base is in Europe. European users consistently report slow application performance, even though server-side metrics show low latency. Network traces reveal high latency between Europe and North America. In another case, two microservices, `Service A` and `Service B`, are deployed in different virtual networks within the same cloud region. A misconfigured network security group or routing table causes traffic between them to be routed through an on-premise data center, introducing significant latency. Network latency is a fundamental constraint in distributed systems. While it cannot be eliminated, it can be mitigated through strategies like content delivery networks (CDNs), edge computing, optimizing network paths, and designing applications to be less sensitive to latency.
