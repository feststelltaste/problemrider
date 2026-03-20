---
title: Responsive Design
description: Design of the user interface that automatically adapts to different screen sizes and device types
category:
- Requirements
- Architecture
quality_tactics_url: https://qualitytactics.de/en/usability/responsive-design/
problems:
- poor-user-experience-ux-design
- user-frustration
- competitive-disadvantage
- negative-user-feedback
- feature-gaps
- high-client-side-resource-consumption
- customer-dissatisfaction
layout: solution
---

## How to Apply ◆

> Legacy systems built for fixed-resolution desktop monitors break on tablets, phones, and modern high-resolution displays. Responsive design ensures the interface adapts to any screen size.

- Replace fixed-width layouts with fluid grids that use relative units such as percentages and viewport units instead of hardcoded pixel widths. Legacy table-based layouts are particularly problematic and should be converted to CSS-based layouts.
- Define responsive breakpoints for the screen sizes most commonly used by the application's users. Test the application at each breakpoint to ensure layouts adapt gracefully.
- Adapt data tables for smaller screens using techniques such as horizontal scrolling, column prioritization that hides less important columns on small screens, or card-based layouts that stack table rows vertically.
- Make touch targets large enough for mobile interaction. Legacy buttons and links designed for mouse cursors are often too small and too close together for finger taps.
- Use responsive images that load appropriate sizes for different screen densities and viewport sizes, reducing bandwidth usage on mobile connections.
- Test on actual devices in addition to browser emulators. Browser emulators miss touch behavior, performance characteristics, and rendering differences on real mobile hardware.

## Tradeoffs ⇄

> Responsive design makes the application usable across all devices, but retrofitting it into a legacy system with a fixed-width layout requires significant frontend work.

**Benefits:**

- Enables users to access the system from any device, including tablets and phones that are increasingly used in the field, in meetings, and for quick lookups.
- Eliminates the need to build and maintain separate mobile applications for basic system access.
- Improves the desktop experience on modern high-resolution and ultra-wide monitors by using available screen space effectively.
- Closes feature gaps compared to competitors who offer responsive or mobile experiences.

**Costs and Risks:**

- Converting a legacy fixed-width layout to responsive design can require reworking every page template and component in the system.
- Complex legacy interfaces with dense data grids and multi-panel layouts are particularly difficult to make responsive without losing functionality.
- Responsive design increases CSS and testing complexity because every screen must look correct at multiple sizes.
- Legacy JavaScript that relies on fixed pixel positions for dropdown menus, popups, and drag-and-drop may break when the layout reflows.

## How It Could Be

> Users increasingly expect to access work systems from multiple devices, and legacy systems that fail to adapt are perceived as outdated and frustrating.

A legacy field service management system requires technicians to use company-issued laptops to check their schedules and access work orders. Technicians find the laptops cumbersome to carry and slow to boot in the field, so they frequently call the office to ask dispatchers to read them work order details. The team implements responsive design for the scheduling and work order screens, using a column-priority pattern for the work order table that shows only essential columns on phone screens and all columns on desktop. Navigation switches from a horizontal toolbar to a hamburger menu on small screens, and touch targets are enlarged for mobile use. Technicians begin using their personal phones to check schedules and review work orders in the field, reducing calls to the dispatch office and improving response times. The responsive implementation costs a fraction of what a native mobile application would have required.
