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
- [Brittle Codebase](brittle-codebase.md)
<br/>  A fragile codebase with poor structure makes it easy for changes to inadvertently break existing functionality.
- [Inadequate Code Reviews](inadequate-code-reviews.md)
<br/>  Poor code reviews fail to identify changes that could break existing functionality before they are merged.
- [Bloated Class](bloated-class.md)
<br/>  Modifying one part of a bloated class frequently breaks unrelated functionality within the same class.
- [Breaking Changes](breaking-changes.md)
<br/>  Previously working client integrations start exhibiting bugs after API changes break their assumptions.
- [Change Management Chaos](change-management-chaos.md)
<br/>  Changes deployed without impact assessment frequently break existing functionality that was previously working.
- [Difficult to Test Code](difficult-to-test-code.md)
<br/>  Without tests to catch regressions, previously fixed bugs resurface when code is modified.
- [Global State and Side Effects](global-state-and-side-effects.md)
<br/>  Changes to global state in one area break existing functionality elsewhere because the dependencies are not apparent.
- [Hidden Dependencies](hidden-dependencies.md)
<br/>  Modifications inadvertently break functionality in components that depend on hidden assumptions or undocumented interactions.
- [Hidden Side Effects](hidden-side-effects.md)
<br/>  Refactoring or reusing functions with hidden side effects inadvertently breaks functionality that depended on those side effects.
- [Inadequate Integration Tests](inadequate-integration-tests.md)
<br/>  Without integration tests, changes to one component can silently break interactions with other components.
- [Incomplete Knowledge](incomplete-knowledge.md)
<br/>  Changes made without awareness of all affected locations inadvertently break functionality in unknown parts of the system.
- [Review Process Breakdown](insufficient-code-review.md)
<br/>  Reviews that miss side effects and coupling issues lead to regression bugs when code changes.
- [Quality Blind Spots](insufficient-testing.md)
<br/>  Insufficient test coverage means changes frequently break existing functionality without detection before release.
- [Legacy Code Without Tests](legacy-code-without-tests.md)
<br/>  Without automated tests to catch regressions, changes frequently break previously working functionality.
- [Long-Lived Feature Branches](long-lived-feature-branches.md)
<br/>  Large merges from long-lived branches introduce many changes at once, increasing the chance of subtle regressions.
- [Lower Code Quality](lower-code-quality.md)
<br/>  Code written without proper care, testing, or design is more fragile and likely to cause regressions when modified.
- [Feedback Isolation](no-continuous-feedback-loop.md)
<br/>  Late-stage changes driven by delayed feedback require rushed modifications that introduce regressions in previously working features.
- [Partial Bug Fixes](partial-bug-fixes.md)
<br/>  Bugs that were supposedly fixed reappear in different contexts because the fix was only applied to some instances of the duplicated code.
- [Poor Domain Model](poor-domain-model.md)
<br/>  Scattered business logic means changes in one area inadvertently break business rules enforced elsewhere.
- [Poor Test Coverage](poor-test-coverage.md)
<br/>  Lack of automated tests means regressions are not caught during development, appearing later in production.
- [Rapid Prototyping Becoming Production](rapid-prototyping-becoming-production.md)
<br/>  Prototype code without proper testing and structure leads to frequent regressions when changes are made.
- [Rapid System Changes](rapid-system-changes.md)
<br/>  Frequent changes without adequate testing time lead to inadvertent breakage of previously working functionality.
- [Reduced Code Submission Frequency](reduced-code-submission-frequency.md)
<br/>  Large, complex changes submitted infrequently are more likely to introduce regressions that are difficult to isolate.
- [Review Process Breakdown](review-process-breakdown.md)
<br/>  Reviews that don't examine code logic thoroughly miss regressions that break previously working functionality.
- [Ripple Effect of Changes](ripple-effect-of-changes.md)
<br/>  Changes that ripple across components frequently introduce regressions in areas that developers didn't realize were affected.
- [Rushed Approvals](rushed-approvals.md)
<br/>  Reviewers who don't carefully examine changes miss regressions that break existing functionality.
- [Superficial Code Reviews](superficial-code-reviews.md)
<br/>  Without deep review of logic changes, regressions slip through and break previously working functionality.
- [Synchronization Problems](synchronization-problems.md)
<br/>  Updating one instance of duplicated logic without updating others causes regressions in the unchanged locations.
- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  Tight coupling means changes in one component can silently break functionality in dependent components, causing regressions.
- [Unpredictable System Behavior](unpredictable-system-behavior.md)
<br/>  Hidden dependencies cause changes to break seemingly unrelated functionality, manifesting as regression bugs.

## Detection Methods ○
- **Automated Regression Test Suites:** Comprehensive automated tests that verify existing functionality after every change
- **User Acceptance Testing:** Systematic testing of key user workflows before releases
- **Production Monitoring:** Real-time monitoring of system behavior to catch regressions quickly
- **A/B Testing:** Gradual rollouts that can detect regressions before full deployment
- **Bug Categorization:** Track and categorize bugs to identify patterns of regression issues

## Examples

A team adds a new feature to their shopping cart that allows users to save items for later. During implementation, they modify the cart persistence logic to support the new functionality. After deployment, users discover that their cart contents are no longer preserved when they log out and log back in—a core feature that had worked perfectly for years. The regression occurred because the new "save for later" feature changed the data structure used to store cart items, but the existing cart loading logic wasn't updated to handle the new format. The automated tests didn't catch this because they only tested the happy path of the new feature, not the existing cart functionality. Another example involves a banking application where a security patch to prevent SQL injection inadvertently breaks the transaction history display for accounts with certain special characters in their names, causing customer service to be flooded with calls from users who can't access their transaction history.
