---
title: Image and Asset Optimization
description: Optimizing images, fonts, and static assets for smaller payloads and
  faster loads
category:
- Performance
problems:
- slow-application-performance
- high-client-side-resource-consumption
- inefficient-frontend-code
- high-resource-utilization-on-client
- gradual-performance-degradation
layout: solution
related_solutions:
- slug: tree-shaking
  similarity: 0.8
- slug: code-splitting
  similarity: 0.8
- slug: compression
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.75
- slug: api-calls-optimization
  similarity: 0.75
- slug: performance-optimization
  similarity: 0.75
---

## Description

Image and asset optimization reduces the size of the images, fonts, and other static files a legacy web application delivers to the browser, through techniques such as converting to modern compressed formats, generating multiple resolutions for responsive delivery, subsetting fonts to only the characters actually used, and deferring the loading of anything not immediately visible. Legacy web applications frequently accumulated these assets over many years without any optimization pipeline in place — full-resolution photographs served unchanged to mobile devices, complete font families loaded for a handful of glyphs, images fetched eagerly regardless of whether they are ever scrolled into view — because asset delivery was not treated as a first-class performance concern at the time the code was written. The effect compounds on slow or mobile connections, where a page assembled from years of unoptimized assets can weigh tens of megabytes and take many seconds to become usable, directly driving the kind of gradual performance degradation and high client-side resource consumption that make an aging system feel sluggish even when its backend logic is unchanged. Automating this optimization within the build pipeline, rather than relying on discipline at upload time, ensures that newly added assets do not silently reintroduce the same problem the effort was meant to solve. Because older browsers still in a legacy system's user base may not support the newest formats, optimization work typically has to include a fallback path, and aggressive compression must be tuned carefully to avoid trading load time for visibly degraded quality.

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Audit existing assets to identify oversized images, unoptimized fonts, and unnecessary static files
- Convert images to modern formats (WebP, AVIF) with appropriate fallbacks for older browsers
- Implement responsive images using srcset to serve appropriately sized images for each device
- Subset web fonts to include only the character sets actually used in the application
- Set appropriate cache headers for static assets and use content-hashed filenames for cache busting
- Implement lazy loading for images and assets that are below the fold or not immediately visible
- Automate asset optimization in the build pipeline so new assets are always optimized before deployment

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces page load times, especially on slow mobile connections
- Decreases bandwidth costs for both the server and the end user
- Improves Core Web Vitals scores, which affect search engine ranking
- Reduces memory consumption on client devices

**Costs and Risks:**
- Modern image formats may not be supported by all browsers used by the legacy application's user base
- Aggressive compression can degrade visual quality unacceptably
- Build pipeline changes may require tooling updates in the legacy project
- Responsive image implementation adds HTML complexity

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy real estate listing platform served full-resolution photographs (averaging 3 MB each) directly to all devices, including mobile phones. A typical listing page with 20 images weighed over 60 MB. The team implemented an image processing pipeline that generated WebP variants at multiple resolutions, served the appropriate size based on the device viewport, and lazy-loaded images below the fold. The median listing page weight dropped from 60 MB to 4 MB. Mobile page load time improved from 12 seconds to 2.5 seconds on a typical 4G connection, and the company's monthly CDN bandwidth costs decreased substantially.
