---
title: Regression Bugs
description: New features or fixes inadvertently break existing functionality that
  was previously working correctly.
category:
- Code
- Process
- Testing
related_problems:
- slug: breaking-changes
  similarity: 0.6
- slug: partial-bug-fixes
  similarity: 0.6
- slug: high-bug-introduction-rate
  similarity: 0.6
- slug: increased-bug-count
  similarity: 0.55
- slug: delayed-bug-fixes
  similarity: 0.55
- slug: increasing-brittleness
  similarity: 0.55
layout: problem
---

## Description

Regression bugs are defects that occur when previously working functionality breaks due to new code changes, feature additions, or bug fixes. These bugs represent a significant threat to software quality because they erode user trust and can reintroduce problems that were thought to be resolved. Regression bugs are particularly problematic because they often go undetected until users encounter them in production, and they indicate fundamental issues with testing practices and code maintainability.

## Indicators ⟡
- Users report that features that used to work are now broken
- Previously passing tests start failing after new deployments
- Customer support receives complaints about functionality that worked in previous versions
- Quality assurance frequently discovers that fixing one bug introduces another
- The team regularly discusses whether changes might "break something else"

## Symptoms ▲

- [Increased Bug Count](increased-bug-count.md)
<br/>  Regression bugs add to the total bug count as previously fixed issues resurface alongside new defects.
- [Customer Dissatisfaction](customer-dissatisfaction.md)
<br/>  Users lose trust when features they relied on break after updates, leading to frustration and dissatisfaction.
- [Refactoring Avoidance](refactoring-avoidance.md)
<br/>  Frequent regressions reinforce fear of making changes, leading teams to avoid refactoring or modifying code.
- [Fear of Change](fear-of-change.md)
<br/>  Repeated experiences of changes breaking existing functionality creates a culture of fear around code modifications.
- [Constant Firefighting](constant-firefighting.md)
<br/>  Regressions in production require urgent fixes, pulling the team into reactive firefighting mode.
## Causes ▼

- [Test Debt](test-debt.md)
<br/>  Insufficient test coverage fails to catch regressions before deployment, allowing them to reach production.
- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  Tightly coupled components mean changes in one area unexpectedly affect seemingly unrelated functionality.
- [Brittle Codebase](brittle-codebase.md)
<br/>  A fragile codebase with poor structure makes it easy for changes to inadvertently break existing functionality.
- [Inadequate Code Reviews](inadequate-code-reviews.md)
<br/>  Poor code reviews fail to identify changes that could break existing functionality before they are merged.
## Detection Methods ○
- **Automated Regression Test Suites:** Comprehensive automated tests that verify existing functionality after every change
- **User Acceptance Testing:** Systematic testing of key user workflows before releases
- **Production Monitoring:** Real-time monitoring of system behavior to catch regressions quickly
- **A/B Testing:** Gradual rollouts that can detect regressions before full deployment
- **Bug Categorization:** Track and categorize bugs to identify patterns of regression issues

## Examples

A team adds a new feature to their shopping cart that allows users to save items for later. During implementation, they modify the cart persistence logic to support the new functionality. After deployment, users discover that their cart contents are no longer preserved when they log out and log back in—a core feature that had worked perfectly for years. The regression occurred because the new "save for later" feature changed the data structure used to store cart items, but the existing cart loading logic wasn't updated to handle the new format. The automated tests didn't catch this because they only tested the happy path of the new feature, not the existing cart functionality. Another example involves a banking application where a security patch to prevent SQL injection inadvertently breaks the transaction history display for accounts with certain special characters in their names, causing customer service to be flooded with calls from users who can't access their transaction history.
