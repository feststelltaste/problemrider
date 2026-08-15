---
title: Feedback
description: Provide visual or acoustic confirmations for user interactions
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/feedback/
problems:
- user-confusion
- user-frustration
- poor-user-experience-ux-design
- negative-user-feedback
- user-trust-erosion
- increased-error-rates
- unpredictable-system-behavior
layout: solution
related_solutions:
- slug: feedback-mechanisms
  similarity: 0.75
- slug: confirmation-dialogs
  similarity: 0.7
- slug: auto-save
  similarity: 0.7
- slug: intuitive-navigation
  similarity: 0.7
- slug: visual-hierarchy
  similarity: 0.7
- slug: real-time-input-validation
  similarity: 0.7
---

## Description

Feedback gives visible or audible confirmation that a user's action — a click, a submission, a command — actually registered, closing the gap left by legacy interfaces that silently complete or silently fail an operation with no acknowledgment at all. That silence is what drives users to click "save" repeatedly or contact support just to confirm something worked, since a system that gives no signal is indistinguishable from one that is broken. Confirmations, progress indicators for anything taking more than a second, and inline validation as users type all close that gap, though feedback that contradicts the actual system state — a "saved successfully" message on a save that silently failed — is worse than saying nothing at all.

## How to Apply ◆

> Legacy systems often provide no visible feedback for user actions, leaving users uncertain about whether their click, submission, or command was registered. Proper feedback closes this communication gap.

- Add immediate visual feedback for every user interaction: buttons should show a pressed state, form submissions should display a success or processing message, and navigation actions should show loading indicators.
- Implement status messages for completed operations that confirm what was done, such as "Record saved successfully" or "3 items deleted." Legacy systems that silently complete operations leave users unsure whether anything happened.
- Show progress indicators for operations that take more than one second. Use determinate progress bars when the completion percentage is known and indeterminate spinners when it is not.
- Provide immediate validation feedback for form inputs as users type or after they leave a field, rather than waiting until the entire form is submitted to reveal errors.
- Use animation sparingly to draw attention to state changes, such as a newly added item appearing with a brief highlight effect in a list.
- Ensure feedback is accessible by not relying solely on color changes. Combine visual indicators with text messages and ARIA live regions for screen reader users.

## Tradeoffs ⇄

> Clear feedback builds user confidence and reduces errors, but requires attention to every interaction point in the system.

**Benefits:**

- Eliminates user uncertainty about whether their actions were registered, directly reducing frustration and double-submissions in legacy systems.
- Reduces error rates because users receive immediate confirmation or correction rather than discovering problems much later.
- Builds user trust by making the system feel responsive and predictable rather than opaque and unreliable.
- Decreases support requests from users who are unsure whether an operation succeeded and contact support to verify.

**Costs and Risks:**

- Implementing comprehensive feedback across a large legacy system requires touching many screens and interaction points, which is labor-intensive.
- Excessive or intrusive feedback, such as pop-up notifications for routine actions, can annoy users and slow them down.
- Feedback that contradicts the actual system state, such as showing "saved successfully" when the save actually failed silently, is worse than no feedback at all.
- Acoustic feedback can be disruptive in shared workspaces and should always be optional.

## How It Could Be

> Lack of feedback is a defining characteristic of many legacy systems, and adding it can transform the user experience with relatively modest effort.

A legacy order management system processes form submissions by reloading the entire page after saving. If the save is successful, the page simply reloads with the same data and no indication that anything happened. If the save fails due to a validation error, the page reloads with the error displayed at the top, but the user's scroll position is lost and they must scroll through a long form to find the problematic field. The team adds inline save status notifications that appear near the save button, confirming success with a brief green message or highlighting the specific fields with errors and scrolling to the first one automatically. Users report that the system "finally tells them what happened," and the number of duplicate submissions caused by users clicking the save button multiple times drops significantly.
