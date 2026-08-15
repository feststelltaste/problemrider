---
title: Virtualized Lists
description: Efficient display of large data lists through virtual scroll areas
category:
- Performance
- Code
problems:
- slow-response-times-for-lists
- high-client-side-resource-consumption
- slow-application-performance
- memory-leaks
- high-resource-utilization-on-client
- inefficient-frontend-code
layout: solution
related_solutions:
- slug: pagination
  similarity: 0.7
- slug: performance-optimization
  similarity: 0.65
- slug: lazy-evaluation
  similarity: 0.65
- slug: lazy-loading
  similarity: 0.65
- slug: image-and-asset-optimization
  similarity: 0.6
- slug: predictive-loading
  similarity: 0.6
---

## Description

Virtualized lists render only the rows currently visible within a scroll viewport, plus a small buffer, and recycle the same limited set of DOM elements as the user scrolls, instead of creating one DOM element for every item in a dataset that may contain tens of thousands of rows. Many legacy frontend components predate this technique and simply render every row of a table or list unconditionally, an approach that scales acceptably for small datasets but degrades catastrophically as the underlying data grows — a pattern common in legacy systems that were built when data volumes were far smaller and nobody anticipated the dataset eventually reaching the size it has today. The performance cost is not a rare edge case at that point but a routine, reproducible freeze on every page load, since the browser has to construct, lay out, and eventually garbage-collect an enormous number of DOM nodes for a view where the user can only ever look at a handful of them at once. Replacing the naive rendering with a virtualization library restores responsiveness by keeping the DOM element count bounded and roughly constant regardless of dataset size, at the cost of added rendering complexity — particularly for variable-height rows — and the loss of certain browser-native behaviors like in-page text search, which typically has to be replaced with an explicit server-side search feature to compensate.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify list or table components that render hundreds or thousands of DOM elements simultaneously
- Replace traditional list rendering with a virtualization library (react-window, react-virtualized, Angular CDK Virtual Scroll, or similar)
- Render only the visible rows plus a small buffer, recycling DOM elements as the user scrolls
- Calculate row heights accurately (fixed or variable) to maintain correct scroll position and scrollbar behavior
- Combine virtualization with server-side pagination so the client never needs to hold the full dataset in memory
- Handle edge cases: keyboard navigation, screen readers, and search-within-list functionality

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Maintains smooth scrolling performance regardless of list size
- Dramatically reduces DOM element count, lowering memory consumption and improving rendering speed
- Enables displaying datasets of tens of thousands of items that would be impossible to render otherwise
- Reduces garbage collection pressure from creating and destroying DOM nodes

**Costs and Risks:**
- Adds complexity to the rendering logic, especially for variable-height rows
- Accessibility can suffer if screen readers cannot access off-screen elements
- Search (Ctrl+F) within the browser does not work for items not currently rendered
- Scroll position management becomes complex when list items are inserted, removed, or resized dynamically
- Integration with legacy DOM-manipulating code may conflict with the virtualization library's assumptions

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy inventory management application rendered all 50,000 product SKUs in a single HTML table, causing the browser to freeze for several seconds during initial render and consuming over 1 GB of memory. The team replaced the table with react-window, rendering only the 30 visible rows plus a 10-row buffer in each direction. Initial render time dropped from 8 seconds to 50 milliseconds, and memory consumption for the list dropped to under 10 MB. The team also added server-side search and filtering so users could find specific SKUs without scrolling through the entire list, compensating for the loss of browser-native text search.
