---
title: Understandable Error Messages
description: Provision of clear, context-related error messages in the event of problems
category:
- Requirements
- Code
quality_tactics_url: https://qualitytactics.de/en/usability/understandable-error-messages/
problems:
- user-confusion
- user-frustration
- poor-user-experience-ux-design
- inadequate-error-handling
- negative-user-feedback
- increased-customer-support-load
- increased-error-rates
- user-trust-erosion
layout: solution
---

## How to Apply ◆

> Legacy systems often display raw technical error messages, stack traces, or cryptic error codes that are meaningless to end users. Understandable error messages explain problems in human terms and guide recovery.

- Audit all error messages in the legacy system and categorize them by severity, frequency, and user impact. Prioritize rewriting the most frequent and most confusing messages first.
- Structure each error message with three components: what happened, why it happened (if known), and what the user can do about it. This pattern ensures every message is actionable.
- Replace technical identifiers and codes with plain language. If error codes must be retained for support purposes, display them in a secondary position such as "Error code: 4021" appended to the human-readable message.
- Position error messages next to the element that caused the error rather than in a generic notification area. For form validation, highlight the specific field and place the message adjacent to it.
- Differentiate error severity visually: use distinct styling for validation warnings, user-recoverable errors, and system errors that require administrator intervention.
- Log the technical details of errors server-side for debugging while showing only user-relevant information in the UI. Users should never see stack traces, SQL errors, or internal exception messages.

## Tradeoffs ⇄

> Clear error messages transform frustrating dead ends into recoverable situations, but require investment in rewriting messages and maintaining them.

**Benefits:**

- Enables users to resolve errors independently rather than contacting support, directly reducing support ticket volume.
- Reduces user frustration and trust erosion caused by confronting incomprehensible technical messages.
- Decreases error rates by helping users understand what went wrong and how to correct their input or approach.
- Eliminates the security risk of exposing internal system details through technical error messages.

**Costs and Risks:**

- Rewriting hundreds of error messages in a large legacy system requires time and collaboration between developers who understand the errors and writers who can communicate clearly.
- Overly simplified error messages that hide relevant details can make it harder for support staff to diagnose issues when users do contact them.
- Error messages must be updated when system behavior changes, or they will provide incorrect guidance.
- Internationalizing error messages adds translation effort for every supported language.

## Examples

> Cryptic error messages are among the most universally frustrating aspects of legacy systems and one of the easiest to improve incrementally.

A legacy inventory management system displays raw database constraint violation messages when users attempt operations that conflict with business rules. A message like "ORA-02292: integrity constraint (INV.FK_ITEM_WAREHOUSE) violated - child record found" appears when a warehouse manager tries to deactivate a warehouse that still contains inventory. The manager has no idea what the message means and calls IT support. The team maps the fifty most common database errors to user-friendly messages. The constraint violation message becomes "This warehouse cannot be deactivated because it still contains active inventory items. Please transfer or write off all items before deactivating." and includes a link to the inventory transfer screen. Support calls related to error messages decrease dramatically, and warehouse managers report feeling more confident managing warehouse configurations because they understand the system's constraints.
