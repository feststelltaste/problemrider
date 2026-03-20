---
title: Mobile First Design
description: The design of applications is primarily done for mobile devices
category:
- Requirements
- Architecture
quality_tactics_url: https://qualitytactics.de/en/usability/mobile-first-design/
problems:
- poor-user-experience-ux-design
- user-frustration
- competitive-disadvantage
- feature-gaps
- negative-user-feedback
- high-client-side-resource-consumption
- customer-dissatisfaction
layout: solution
---

## How to Apply ◆

> Legacy systems were typically designed for desktop use with large screens and mouse input. As mobile usage grows, these systems become increasingly unusable on smaller devices, creating pressure for mobile-first redesign.

- Assess which workflows users need to perform on mobile devices. Not all legacy functionality needs to be mobile-optimized; focus on the tasks users actually perform while away from their desks.
- Design the mobile experience first, then progressively enhance it for larger screens. This forces prioritization of the most essential content and actions rather than trying to shrink a desktop layout.
- Replace hover-dependent interactions with touch-friendly alternatives. Legacy interfaces that rely on mouseover tooltips, hover menus, and right-click context menus are unusable on touch devices.
- Optimize touch target sizes for mobile. Buttons and interactive elements should be at least 44 by 44 pixels to be reliably tappable, significantly larger than many legacy interface elements.
- Minimize data transfer for mobile connections by lazy-loading images, paginating large data sets, and compressing API responses. Legacy systems often send entire data sets to the client regardless of what the user needs.
- Use responsive breakpoints to adapt layouts rather than maintaining separate mobile and desktop codebases, which doubles the maintenance burden.

## Tradeoffs ⇄

> Mobile-first design ensures the system works on any device, but fundamentally challenges the design assumptions of desktop-oriented legacy systems.

**Benefits:**

- Makes the system accessible to users in the field, in meetings, and during travel, addressing a growing expectation for mobile access.
- Forces simplification of complex interfaces because mobile constraints require prioritization of essential functionality.
- Improves performance for all users because mobile optimization techniques like lazy loading and data compression benefit desktop users too.
- Closes competitive gaps with modern alternatives that offer mobile experiences out of the box.

**Costs and Risks:**

- Redesigning a legacy system for mobile-first is a major undertaking that may require rethinking the entire frontend architecture.
- Some legacy workflows involving complex data entry, multi-column tables, or detailed charts may not translate well to small screens without significant rethinking.
- Supporting both mobile and desktop in a legacy codebase increases testing complexity and the number of layouts to maintain.
- Mobile-first design may require a modern responsive frontend framework, which conflicts with legacy frontend technologies.

## Examples

> Field workers using legacy systems on mobile devices often resort to paper-based workarounds because the system is unusable on their phones.

A legacy facilities management system requires maintenance technicians to log completed work orders using the desktop application after returning to the office. Technicians carry paper forms in the field and enter the data hours later, leading to incomplete records, forgotten details, and delayed billing. The team builds a mobile-first interface for the work order completion workflow, focusing on the five fields technicians need in the field: status update, time spent, parts used, a notes field, and photo upload. The mobile interface uses large touch-friendly controls, works offline with sync-when-connected capability, and loads quickly on cellular connections. Technicians begin completing work orders on-site immediately after finishing each job. Data quality improves because details are captured while fresh, and billing cycles shorten because completed work orders no longer wait for end-of-day data entry.
