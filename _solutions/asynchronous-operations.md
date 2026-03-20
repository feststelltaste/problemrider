---
title: Asynchronous Operations
description: Execution of time-intensive operations in the background without blocking the user interface
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/usability/asynchronous-operations/
problems:
- slow-application-performance
- user-frustration
- poor-user-experience-ux-design
- slow-response-times-for-lists
- high-client-side-resource-consumption
- thread-pool-exhaustion
- external-service-delays
- negative-user-feedback
layout: solution
---

## How to Apply ◆

> Legacy systems often execute long-running operations synchronously, freezing the user interface and frustrating users. Moving these operations to background processing improves perceived responsiveness.

- Identify synchronous operations that take more than one or two seconds to complete, such as report generation, data exports, batch processing, and external service calls. These are the primary candidates for asynchronous conversion.
- Implement background job processing using a message queue or task queue. Replace synchronous request-response patterns with a submit-and-poll or submit-and-notify approach.
- Add progress indicators and status notifications so users know their operation is being processed. Legacy systems that simply show a spinner or freeze the screen provide no useful feedback.
- Implement proper error handling for background operations, including retry mechanisms and clear error notifications. Users must be informed when a background operation fails, not left wondering.
- Use optimistic UI updates where appropriate: show the expected result immediately while the actual operation completes in the background, rolling back only if the operation fails.
- Ensure that the UI remains fully interactive while background operations are running. Users should be able to continue working on other tasks without waiting.

## Tradeoffs ⇄

> Asynchronous operations dramatically improve user experience but introduce complexity in state management and error handling.

**Benefits:**

- Eliminates UI freezing during long-running operations, directly addressing user frustration caused by unresponsive legacy interfaces.
- Improves perceived performance even when actual processing time remains the same, because users can continue working.
- Reduces thread pool exhaustion on the server by offloading long-running work to background workers instead of holding HTTP request threads.
- Enables better handling of external service delays by decoupling the user interaction from the downstream processing.

**Costs and Risks:**

- Introduces complexity in tracking operation state, handling failures, and ensuring consistency between the UI state and the actual backend state.
- Requires infrastructure for background job processing such as message queues and worker processes that the legacy system may not currently have.
- Testing asynchronous flows is more complex than testing synchronous ones, requiring consideration of timing, race conditions, and failure scenarios.
- Users accustomed to synchronous feedback may initially be confused by the change in interaction pattern and need clear communication about what is happening.

## Examples

> Background processing can transform the most frustrating aspects of a legacy system into responsive, user-friendly experiences.

A legacy ERP system generates monthly financial reports synchronously, locking the user's session for up to fifteen minutes while the report compiles. Users have learned to start the report and leave for coffee, but if the session times out, they must restart the entire process. The team introduces a background report generation service: users submit report requests and receive a notification when the report is ready for download. The report generation itself takes the same amount of time, but users can continue entering invoices and managing purchase orders while waiting. Session timeout issues disappear entirely, and the support team stops receiving tickets about lost reports.
