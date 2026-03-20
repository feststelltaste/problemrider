---
title: Optimistic UI Updates
description: Reduce perceived latency by updating the interface before server confirmation
category:
- Performance
- Code
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/optimistic-ui-updates
problems:
- slow-application-performance
- high-api-latency
- poor-user-experience-ux-design
- user-frustration
- external-service-delays
- network-latency
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify user interactions where the server response is predictable and the success rate is high (e.g., toggling a setting, adding an item to a list)
- Update the UI state immediately upon user action, before the server request completes
- Implement rollback logic that reverts the UI to its previous state if the server returns an error
- Display subtle, non-blocking indicators to communicate that the change is being persisted
- Start with low-risk operations and expand to more complex interactions as the team gains confidence in the pattern
- Ensure idempotency on the server side to handle retries gracefully when rollbacks and resubmissions occur

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Dramatically improves perceived responsiveness without backend performance changes
- Reduces user frustration caused by waiting for server round-trips
- Makes the application feel modern and responsive even when backed by a slow legacy API
- Can be applied incrementally to specific interactions without rewriting the entire frontend

**Costs and Risks:**
- Rollback logic adds complexity and must handle edge cases such as concurrent updates
- Users may be confused if an action appears to succeed but is later reverted
- Requires server-side idempotency guarantees that legacy APIs may not provide
- Increases the gap between displayed state and actual server state, which can cause subtle inconsistencies
- Not suitable for operations where failure is common or consequences are significant

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A project management tool built on a legacy REST API had response times averaging 800 milliseconds for status updates. Users frequently double-clicked or navigated away before updates completed, causing confusion and duplicate requests. The team implemented optimistic updates for task status changes, immediately reflecting the new status in the UI while the API call proceeded in the background. On the rare occasions when the server rejected the update, a toast notification informed the user and the status reverted. This change reduced perceived latency to near-zero and eliminated duplicate submission complaints, all without modifying the legacy backend.
