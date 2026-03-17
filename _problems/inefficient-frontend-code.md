---
title: Inefficient Frontend Code
description: Unoptimized JavaScript, excessive DOM manipulation, or complex CSS animations
  that are computationally expensive.
category:
- Code
- Performance
related_problems:
- slug: high-client-side-resource-consumption
  similarity: 0.8
- slug: high-resource-utilization-on-client
  similarity: 0.75
- slug: inefficient-code
  similarity: 0.75
- slug: slow-application-performance
  similarity: 0.7
- slug: imperative-data-fetching-logic
  similarity: 0.6
- slug: slow-database-queries
  similarity: 0.6
layout: problem
---

## Description
Inefficient frontend code can have a significant impact on the user experience. This can manifest as a slow-loading page, a sluggish user interface, or a high level of resource consumption on the client's machine. Common causes of inefficient frontend code include large, unoptimized assets, excessive DOM manipulation, and a lack of proper caching. A focus on frontend performance is essential for creating a fast and responsive user experience.

## Indicators ⟡
- The application is slow, even on a powerful device.
- The application is draining the battery on your mobile device.
- Your computer's fan is running at high speed when you use your application.
- You are getting complaints from users about slow performance.

## Symptoms ▲

- [Slow Application Performance](slow-application-performance.md)
<br/>  Unoptimized frontend code causes sluggish page loads and UI interactions visible to end users.
- [High Client-Side Resource Consumption](high-client-side-resource-consumption.md)
<br/>  Inefficient DOM manipulation and unoptimized JavaScript consume excessive CPU and memory on user devices.
- [High Resource Utilization on Client](high-resource-utilization-on-client.md)
<br/>  Computationally expensive frontend operations cause high CPU usage, battery drain, and device heating on client machines.
## Causes ▼

- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers unfamiliar with browser rendering pipelines and performance optimization create inefficient frontend code.
- [Insufficient Code Review](insufficient-code-review.md)
<br/>  Without reviews focused on performance, frontend anti-patterns like excessive DOM manipulation go undetected.
- [Lower Code Quality](lower-code-quality.md)
<br/>  General code quality issues manifest as unoptimized rendering logic, redundant computations, and wasteful resource usage in the frontend.
## Detection Methods ○

- **Browser Developer Tools:** Use the Performance, Memory, and Network tabs in browser developer tools to profile CPU usage, memory consumption, and rendering performance.
- **Lighthouse/PageSpeed Insights:** Use these tools to get automated audits and suggestions for frontend performance.
- **Real User Monitoring (RUM):** RUM tools can collect client-side performance metrics from actual user sessions.
- **Code Review:** Look for common anti-patterns like direct DOM manipulation in loops, complex CSS selectors, or large JavaScript files.
- **Web Vitals:** Monitor Core Web Vitals (LCP, FID, CLS) to understand user experience.

## Examples
A web application displays a large table with thousands of rows. Each time a user sorts or filters the table, the entire DOM is re-rendered, causing a noticeable freeze in the UI. Implementing a virtualized list or table component would significantly improve performance. In another case, a JavaScript function is responsible for updating a counter on the screen every second. Instead of directly updating the text content of a single DOM element, it re-creates the entire element and appends it to the DOM, leading to constant reflows and high CPU usage. With the increasing complexity of web applications and the demand for rich user experiences, efficient frontend code is paramount. This problem is common in applications that have grown organically without a strong focus on performance optimization or where developers lack deep knowledge of browser rendering pipelines.
