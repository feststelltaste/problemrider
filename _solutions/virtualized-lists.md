---
title: Virtualized Lists
description: Efficient display of large data lists through virtual scroll areas
category:
- Performance
- Code
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/virtualized-lists
problems:
- slow-response-times-for-lists
- high-client-side-resource-consumption
- slow-application-performance
- memory-leaks
- high-resource-utilization-on-client
- inefficient-frontend-code
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy inventory management application rendered all 50,000 product SKUs in a single HTML table, causing the browser to freeze for several seconds during initial render and consuming over 1 GB of memory. The team replaced the table with react-window, rendering only the 30 visible rows plus a 10-row buffer in each direction. Initial render time dropped from 8 seconds to 50 milliseconds, and memory consumption for the list dropped to under 10 MB. The team also added server-side search and filtering so users could find specific SKUs without scrolling through the entire list, compensating for the loss of browser-native text search.
