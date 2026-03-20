---
title: Undo and Redo
description: Allowing users to reverse and reapply actions for error recovery and exploration
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/undo-and-redo/
problems:
- user-frustration
- poor-user-experience-ux-design
- user-trust-erosion
- increased-error-rates
- fear-of-change
- negative-user-feedback
- customer-dissatisfaction
layout: solution
---

## How to Apply ◆

> Legacy systems rarely support undo, making every action potentially irreversible and causing users to be cautious, slow, and anxious. Adding undo and redo capability enables confident exploration and quick error recovery.

- Implement an action history stack that records user modifications and allows them to be reversed. Start with the most common and most consequential actions rather than trying to make everything undoable at once.
- Support multi-level undo that allows users to reverse a sequence of actions, not just the most recent one. Display the undo history so users can see what they are reversing.
- Implement redo alongside undo so users who accidentally undo too far can reapply their changes without re-entering them.
- Provide an undo option in success confirmations, such as "Item deleted. [Undo]" with a time-limited window for reversal. This is simpler to implement than full undo history and handles the most common use case.
- For database operations, implement undo through soft deletes, audit trails, or event sourcing rather than attempting to reverse database transactions, which is complex and fragile in legacy systems.
- Communicate the undo capability clearly through keyboard shortcut support (Ctrl+Z / Cmd+Z) and visible undo buttons, so users know the safety net exists.

## Tradeoffs ⇄

> Undo capability fundamentally changes user behavior by removing the fear of making mistakes, but implementing it in a legacy system can be architecturally challenging.

**Benefits:**

- Eliminates the fear of making mistakes that causes users to avoid exploring features or making changes in legacy systems.
- Dramatically reduces the consequences of accidental actions, decreasing the need for data recovery by administrators.
- Enables rapid experimentation because users can try things and easily reverse them if the result is not what they expected.
- Builds user trust by providing a visible safety net that demonstrates the system respects users' ability to change their minds.

**Costs and Risks:**

- Implementing undo in a legacy system with direct database writes and no audit trail requires architectural changes to record reversible actions.
- Some operations are inherently difficult or impossible to undo, such as sending emails, triggering external API calls, or starting physical processes. Clear communication about which actions are undoable is essential.
- Undo history consumes storage and must be bounded to prevent unbounded growth, especially in high-volume systems.
- Multi-user environments create complications: undoing a change that another user has subsequently modified can create conflicts.

## How It Could Be

> The absence of undo in legacy systems creates a culture of cautious, slow interaction where users are afraid to explore or experiment.

A legacy content management system used by a marketing team has no undo capability. Every text change, image replacement, and layout modification is immediately permanent. Team members have developed a habit of copying entire pages to backup folders before making any edits, cluttering the system with hundreds of backup copies. The team implements a version history system that automatically saves a snapshot before each save operation, with a comparison view that shows what changed between versions and a one-click restore option. The team also adds a time-limited undo toast notification that appears after each save with "Changes saved. [Undo]" for quick reversals. Marketing staff stop creating manual backups, the system becomes cleaner, and team members report being more willing to experiment with content and layout changes because they know they can easily revert.
