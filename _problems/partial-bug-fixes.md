---
title: Partial Bug Fixes
description: Issues appear to be resolved but resurface in different contexts because
  the fix was not applied to all instances of the duplicated code.
category:
- Code
related_problems:
- slug: delayed-bug-fixes
  similarity: 0.7
- slug: difficult-code-reuse
  similarity: 0.65
- slug: incomplete-knowledge
  similarity: 0.65
- slug: inconsistent-quality
  similarity: 0.6
- slug: code-duplication
  similarity: 0.6
- slug: synchronization-problems
  similarity: 0.6
solutions:
- definition-of-done
- regression-tests
- root-cause-analysis
layout: problem
---

## Description
Partial bug fixes are a common problem in software systems with a high degree of code duplication. They occur when a bug is fixed in one instance of the duplicated code, but not in all of them. This can lead to a number of problems, including regression bugs, a loss of trust in the system, and a great deal of frustration for the development team. Partial bug fixes are often a sign of a poorly designed system with a high degree of code duplication.

## Indicators ⟡
- The same bug is reported over and over again.
- The team is constantly fixing regression bugs.
- The team is not sure if a bug has been fixed.
- The team is not able to reproduce bugs that are reported by users.

## Symptoms ▲

- [Regression Bugs](regression-bugs.md)
<br/>  Bugs that were supposedly fixed reappear in different contexts because the fix was only applied to some instances of the duplicated code.
- [User Frustration](user-frustration.md)
<br/>  Users experience the same bug repeatedly after being told it was fixed, causing frustration and loss of trust.
- [Inconsistent Behavior](inconsistent-behavior.md)
<br/>  The same business process works correctly in one context but fails in another because the fix was not applied uniformly across duplicated code.
- [Increased Bug Count](increased-bug-count.md)
<br/>  Each partial fix only addresses one instance of the bug while leaving others open, keeping the defect count high.
## Causes ▼

- [Code Duplication](code-duplication.md)
<br/>  Duplicated code is the primary enabler of partial bug fixes, as the same logic exists in multiple places that must all be updated.
- [Incomplete Knowledge](incomplete-knowledge.md)
<br/>  Developers are unaware of all locations where the same logic exists, so they fix the bug in the place they know about but miss other instances.
- [Poor Test Coverage](poor-test-coverage.md)
<br/>  Without comprehensive tests covering all instances of duplicated logic, partial fixes go undetected until users encounter the unfixed instances.
- [Time Pressure](time-pressure.md)
<br/>  Under pressure to resolve bugs quickly, developers fix the reported instance without searching for and fixing all occurrences.
## Detection Methods ○
- **Code Duplication Analysis:** Use static analysis tools to identify duplicated code.
- **Regression Testing:** Use regression testing to verify that bugs that were previously fixed have not reappeared.
- **Code Reviews:** Code reviews are a great way to identify partial bug fixes.
- **Bug Tracking System:** Use a centralized bug tracking system to track the status of bugs.

## Examples
An e-commerce website has a bug in its checkout flow. The bug is fixed in the checkout flow for regular customers, but it is not fixed in the checkout flow for guest customers. As a result, the bug is still present in the system, and it is still affecting users. The problem could have been avoided if the developer who fixed the bug had been aware of the duplicated code and had fixed the bug in both places.
