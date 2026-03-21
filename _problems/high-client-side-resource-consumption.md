---
title: High Client-Side Resource Consumption
description: Client applications consume excessive CPU or memory, leading to sluggish
  performance and poor user experience.
category:
- Performance
related_problems:
- slug: high-resource-utilization-on-client
  similarity: 0.95
- slug: inefficient-frontend-code
  similarity: 0.8
- slug: high-database-resource-utilization
  similarity: 0.7
- slug: slow-application-performance
  similarity: 0.7
- slug: resource-contention
  similarity: 0.7
- slug: memory-leaks
  similarity: 0.65
solutions:
- code-splitting
- lazy-loading
- image-and-asset-optimization
- tree-shaking
- virtualized-lists
- performance-budgets
- compression
- pagination
- progressive-loading
- lazy-evaluation
layout: problem
---

## Description
High client-side resource consumption can lead to a poor user experience. This can manifest as a sluggish user interface, a high level of battery consumption on mobile devices, or a general feeling of unresponsiveness. Common causes of high resource consumption include inefficient JavaScript, large, unoptimized assets, and excessive DOM manipulation. A focus on client-side performance is essential for creating a fast and responsive user experience.

## Indicators ⟡
- Your application is slow, even on a powerful device.
- Your application is draining the battery on your mobile device.
- Your computer's fan is running at high speed when you use your application.
- You are getting complaints from users about slow performance.

## Symptoms ▲

- [Slow Application Performance](slow-application-performance.md)
<br/>  Excessive CPU and memory usage on the client causes the application to feel sluggish and unresponsive.
- [Negative User Feedback](negative-user-feedback.md)
<br/>  Users complain about slow performance, hot devices, and battery drain caused by resource-heavy client applications.
- [User Frustration](user-frustration.md)
<br/>  Users become dissatisfied when the application makes their device slow, hot, or drains battery quickly.
## Causes ▼

- [Inefficient Frontend Code](inefficient-frontend-code.md)
<br/>  Unoptimized JavaScript, excessive DOM manipulation, and complex CSS animations consume excessive client CPU and memory.
- [Memory Leaks](memory-leaks.md)
<br/>  Client-side memory leaks from unreleased DOM elements or event listeners cause continuously growing memory consumption.
- [Improper Event Listener Management](improper-event-listener-management.md)
<br/>  Event listeners that are never removed accumulate over time, consuming memory and CPU resources on the client.
- [Inefficient Code](inefficient-code.md)
<br/>  Computationally expensive client-side code, such as large loops or complex rendering logic, consumes excessive CPU resources.
## Detection Methods ○

- **Browser Developer Tools:** Use the Performance, Memory, and Network tabs in browser developer tools to profile CPU usage, memory consumption, and network activity.
- **Real User Monitoring (RUM):** RUM tools can collect performance metrics from actual user sessions, including CPU and memory usage.
- **Device-Specific Monitoring:** Use tools provided by operating systems (e.g., Activity Monitor on macOS, Task Manager on Windows, Android Studio Profiler, Xcode Instruments) to monitor resource usage.
- **Code Review:** Look for common anti-patterns like large loops, excessive event listeners, or unoptimized rendering logic.

## Examples
A single-page application (SPA) becomes very slow after a user has been interacting with it for a long time. Profiling reveals a memory leak where old DOM elements are not being garbage collected, leading to continuous memory growth. In another case, a website uses a large, unoptimized background video on its homepage. On mobile devices, this causes the browser to consume a significant amount of CPU and battery, making the phone hot and draining the battery quickly. This problem is increasingly common with the rise of complex web applications and mobile apps that run directly on user devices. Optimizing client-side performance is crucial for delivering a smooth and enjoyable user experience.
