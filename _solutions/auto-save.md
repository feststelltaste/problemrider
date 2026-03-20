---
title: Auto-Save
description: Automatically saving user work at regular intervals against data loss
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/auto-save/
problems:
- user-frustration
- poor-user-experience-ux-design
- negative-user-feedback
- customer-dissatisfaction
- user-trust-erosion
- increased-customer-support-load
layout: solution
---

## How to Apply ◆

> Legacy systems frequently rely on explicit manual saves, leading to data loss when sessions time out, browsers crash, or users forget to save. Auto-save protects user work transparently.

- Implement periodic auto-save that captures the current form or document state at regular intervals, typically every thirty to sixty seconds. Store drafts server-side or in local storage as a fallback.
- Add visual indicators that show when data was last auto-saved, giving users confidence that their work is protected without requiring them to think about saving.
- Implement conflict resolution for scenarios where auto-saved data conflicts with changes made by another user or in another session. Display a clear comparison view rather than silently overwriting.
- Add session recovery that detects unsaved work from a previous session and offers to restore it when the user returns. This is critical for legacy systems with aggressive session timeouts.
- Ensure auto-save does not interfere with validation. Save drafts even if the form is in an incomplete or invalid state, deferring validation to the explicit submit action.
- Throttle auto-save requests to avoid overwhelming the server, especially in legacy systems with limited backend capacity.

## Tradeoffs ⇄

> Auto-save eliminates an entire category of user frustration but requires careful implementation to avoid data consistency issues.

**Benefits:**

- Eliminates data loss from session timeouts, browser crashes, and accidental navigation, which are among the most frustrating experiences in legacy systems.
- Reduces support tickets related to lost work, directly decreasing the customer support load.
- Builds user trust by demonstrating that the system protects their work rather than discarding it at the slightest disruption.
- Removes the cognitive burden of remembering to save, allowing users to focus on their actual tasks.

**Costs and Risks:**

- Auto-saving incomplete or invalid data requires careful handling to avoid polluting the database with draft records that are never finalized.
- Increases server load due to frequent save requests, which may strain already resource-constrained legacy backends.
- Conflict resolution between auto-saved drafts and concurrent edits adds complexity, especially in multi-user legacy systems without optimistic locking.
- Users who are accustomed to explicit saves may be confused by auto-save behavior if it is not clearly communicated through the interface.

## How It Could Be

> Data loss is a persistent source of frustration in legacy systems, and auto-save directly addresses the most common scenarios.

A legacy procurement system requires users to fill out multi-page purchase requisition forms. The system has a thirty-minute session timeout, and users frequently lose their work when they pause to gather information from other systems. The team implements auto-save using local storage for the draft state, with server-side persistence every sixty seconds. When a user returns after a session timeout, the system detects the saved draft and offers to restore it. Lost-work support tickets drop from approximately twenty per week to near zero, and user satisfaction scores for the procurement module improve significantly in the next internal survey.
