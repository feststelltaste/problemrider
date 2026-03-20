---
title: Performance Optimization
description: Improving perceived responsiveness through user-facing performance techniques
category:
- Performance
quality_tactics_url: https://qualitytactics.de/en/usability/performance-optimization/
problems:
- slow-application-performance
- user-frustration
- poor-user-experience-ux-design
- slow-response-times-for-lists
- high-client-side-resource-consumption
- negative-user-feedback
- gradual-performance-degradation
- customer-dissatisfaction
layout: solution
---

## How to Apply ◆

> Legacy systems often suffer from slow perceived performance due to full page reloads, unoptimized rendering, and blocking operations. User-facing performance optimization targets what users actually experience rather than raw server metrics.

- Implement skeleton screens and loading placeholders that show the structure of the page immediately while data loads. This makes the application feel faster than showing a blank screen or spinner.
- Add pagination or virtual scrolling to large data sets. Legacy systems that load thousands of rows into the browser simultaneously cause severe rendering delays and excessive memory consumption.
- Defer loading of non-critical content. Apply lazy loading to images, secondary data panels, and below-the-fold content so the primary view renders quickly.
- Optimize the critical rendering path by loading essential CSS and JavaScript first and deferring non-essential resources. Legacy systems often load large monolithic bundles that block rendering.
- Implement client-side caching for data that changes infrequently, reducing the number of server round-trips and improving response times for repeated views.
- Pre-fetch data for likely next actions based on user behavior patterns. If users typically navigate from a list to a detail view, start loading the detail data when the user hovers over or focuses on a list item.

## Tradeoffs ⇄

> Perceived performance improvements make the system feel dramatically faster, but add frontend complexity and may mask underlying backend performance issues.

**Benefits:**

- Dramatically improves user perception of application speed, even when backend processing times remain unchanged.
- Reduces user frustration caused by waiting for slow-loading pages, which is one of the most common complaints about legacy systems.
- Decreases client-side resource consumption through efficient rendering and data loading strategies.
- Can be implemented incrementally on individual screens without requiring a complete frontend rewrite.

**Costs and Risks:**

- Skeleton screens and optimistic updates can mislead users if the actual data takes significantly longer to load or differs from the preview.
- Client-side caching introduces the risk of displaying stale data, requiring careful cache invalidation strategies.
- Optimizing perceived performance may reduce pressure to fix underlying backend performance issues, allowing them to worsen over time.
- Implementing lazy loading and virtual scrolling in legacy frontend frameworks that do not natively support them can be technically challenging.

## Examples

> Users judge application speed by what they see, not by what the server logs say. Optimizing perceived performance can transform a sluggish legacy system without touching the backend.

A legacy customer management system loads a customer list page that retrieves all customer records from the database and renders them in a single HTML table. With over fifty thousand customers, the page takes twelve seconds to load, during which users see a blank white screen. The team implements three changes: virtual scrolling that renders only the visible rows, a skeleton screen that appears instantly while data loads, and paginated API calls that fetch one hundred records at a time. The first meaningful content now appears in under one second, and users can start scrolling and searching immediately while additional data loads in the background. Although the total data transfer is the same, users perceive the system as dramatically faster because they can begin working almost immediately instead of staring at a blank screen.
