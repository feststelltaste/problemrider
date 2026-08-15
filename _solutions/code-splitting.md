---
title: Code Splitting
description: Splitting the application code into smaller chunks
category:
- Performance
- Code
problems:
- slow-application-performance
- high-client-side-resource-consumption
- inefficient-frontend-code
- gradual-performance-degradation
- feature-bloat
- high-resource-utilization-on-client
layout: solution
related_solutions:
- slug: tree-shaking
  similarity: 0.8
- slug: lazy-loading
  similarity: 0.8
- slug: image-and-asset-optimization
  similarity: 0.8
- slug: lazy-evaluation
  similarity: 0.75
- slug: predictive-loading
  similarity: 0.7
- slug: performance-optimization
  similarity: 0.7
---

## Description

Code splitting is a build-time technique that divides an application's compiled output into multiple smaller chunks, loaded on demand rather than shipping the entire application as one monolithic bundle on every page visit. Route-based splitting ensures a given page loads only the code that page actually needs, while dynamic imports defer loading features that are not needed on initial render — admin panels, rarely used tools, modal dialogs — until the moment a user actually reaches them. This matters in legacy single-page applications especially, which frequently accumulated features over years without any attention to bundle size, resulting in every user downloading megabytes of JavaScript for functionality that only a small fraction of them will ever touch, an unnecessary cost that grows worse the longer the legacy application has been extended. Splitting vendor libraries into their own chunk, separate from application code, additionally improves caching, since a vendor bundle that rarely changes does not need to be re-downloaded every time application code is updated. Because it operates at build configuration level, code splitting can typically be introduced without redesigning the application's actual logic, making it a comparatively low-risk performance intervention relative to deeper architectural changes. Its main risk is over-splitting: dividing the bundle into too many small chunks trades one performance problem (a large initial download) for another (excessive numbers of network requests), so the split boundaries need to be tuned against real usage patterns rather than applied mechanically everywhere.

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
