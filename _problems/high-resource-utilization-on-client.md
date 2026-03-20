---
title: High Resource Utilization on Client
description: Client applications may consume excessive CPU or memory, leading to a
  poor user experience, especially on less powerful devices.
category:
- Performance
- Requirements
related_problems:
- slug: high-client-side-resource-consumption
  similarity: 0.95
- slug: inefficient-frontend-code
  similarity: 0.75
- slug: high-database-resource-utilization
  similarity: 0.75
- slug: resource-contention
  similarity: 0.7
- slug: slow-application-performance
  similarity: 0.7
- slug: high-api-latency
  similarity: 0.6
solutions:
- user-centered-design
layout: problem
---

## Description
High resource utilization on the client-side can lead to a poor user experience. This can manifest as a sluggish user interface, a high level of battery consumption on mobile devices, or a general feeling of unresponsiveness. Common causes of high resource utilization include inefficient JavaScript, large, unoptimized assets, and excessive DOM manipulation. A focus on client-side performance is essential for creating a fast and responsive user experience.

## Indicators ⟡
- Your application is slow, even on a powerful device.
- Your application is draining the battery on your mobile device.
- Your computer's fan is running at high speed when you use your application.
- You are getting complaints from users about slow performance.

## Symptoms ▲

- [Slow Application Performance](slow-application-performance.md)
<br/>  Excessive client-side resource consumption causes the application UI to become sluggish and unresponsive for users.
- [Customer Dissatisfaction](customer-dissatisfaction.md)
<br/>  Users experience poor performance, battery drain, and device overheating, leading to frustration and abandonment.
## Causes ▼

- [Inefficient Frontend Code](inefficient-frontend-code.md)
<br/>  Poorly optimized JavaScript, excessive DOM manipulation, and unnecessary re-renders consume excessive client resources.
- [Memory Leaks](memory-leaks.md)
<br/>  Unreleased memory from improperly managed objects and event listeners gradually consumes available client resources.
- [Improper Event Listener Management](improper-event-listener-management.md)
<br/>  Accumulated unremoved event listeners consume memory and execute unnecessary code, increasing CPU and memory usage.
## Detection Methods ○

- **Browser Developer Tools:** Use the Performance, Memory, and Network tabs in browser developer tools to profile client-side activity.
- **Real User Monitoring (RUM):** RUM tools can collect client-side performance metrics from actual users.
- **Device Monitoring Tools:** Use OS-level tools (e.g., Activity Monitor on macOS, Task Manager on Windows, Android Studio Profiler) to monitor CPU and memory usage of the client application.
- **User Feedback:** Pay attention to user complaints about performance, battery life, or device overheating.

## Examples
A complex web application with many interactive elements becomes very slow and causes the user's laptop fan to spin up. Profiling with browser developer tools reveals that a JavaScript function is constantly re-rendering a large part of the DOM in an inefficient loop. In another case, a mobile game has unoptimized textures and models. When played on an older phone, the game frequently lags and causes the device to become very hot, draining the battery quickly. This problem is increasingly common as applications become more feature-rich and run on a wider variety of devices. Optimizing client-side performance is crucial for a good user experience, especially on mobile and lower-end hardware.
