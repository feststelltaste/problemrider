---
title: Code Splitting
description: Splitting the application code into smaller chunks
category:
- Performance
- Code
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/code-splitting
problems:
- slow-application-performance
- high-client-side-resource-consumption
- inefficient-frontend-code
- gradual-performance-degradation
- feature-bloat
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Analyze the application bundle to identify large modules and dependencies that contribute most to initial load size
- Implement route-based splitting so each page loads only the code it needs
- Use dynamic imports for features that are not needed on initial render: modals, admin panels, rarely used tools
- Split vendor libraries into a separate chunk that can be cached independently from application code
- Configure the build tool (Webpack, Vite, esbuild) to set appropriate chunk size limits and naming strategies
- Implement prefetching for code chunks that the user is likely to need next based on navigation patterns
- Monitor real user metrics to verify that splitting improves actual load times

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces initial page load time by loading only the code needed for the current view
- Improves cache efficiency because unchanged chunks are not re-downloaded on updates
- Enables incremental loading that makes the application feel faster to users
- Reduces memory consumption on resource-constrained devices

**Costs and Risks:**
- Adds complexity to the build configuration and module structure
- May introduce loading delays when navigating to new sections that require fetching additional chunks
- Over-splitting creates too many small network requests, which can worsen performance
- Legacy bundling configurations may require significant rework to support code splitting

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy single-page application for an insurance portal loaded a 4.5 MB JavaScript bundle on every page load, including code for agent dashboards, claim submission forms, and reporting charts that most users never accessed. The team introduced route-based code splitting, reducing the initial bundle to 800 KB and loading additional modules on demand. They also split the charting library into a lazy-loaded chunk since only the reporting section used it. Average page load time dropped from 6 seconds to 1.8 seconds on typical connections, and mobile users reported a dramatically improved experience.
