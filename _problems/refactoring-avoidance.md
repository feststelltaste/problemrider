---
title: Refactoring Avoidance
description: The development team actively avoids refactoring the codebase, even when
  they acknowledge it's necessary, due to fear of introducing new bugs.
category:
- Code
- Process
related_problems:
- slug: resistance-to-change
  similarity: 0.75
- slug: fear-of-change
  similarity: 0.7
- slug: maintenance-paralysis
  similarity: 0.7
- slug: fear-of-breaking-changes
  similarity: 0.65
- slug: brittle-codebase
  similarity: 0.6
- slug: high-technical-debt
  similarity: 0.6
layout: problem
---

## Description
Refactoring avoidance is the phenomenon where a development team consistently postpones or avoids improving the internal structure of the code, even when they are aware of its deficiencies. This is often driven by a fear that any change, no matter how well-intentioned, will introduce new bugs or break existing functionality. This avoidance is a self-perpetuating cycle: the longer refactoring is delayed, the more technical debt accumulates, and the riskier any future changes become. It is a clear sign of a fragile and unhealthy codebase.

## Indicators ⟡
- Developers say things like, "Don't touch that code, it's a house of cards."
- The team consistently chooses to add new code rather than modify existing code.
- There is a long list of known technical debt items that never gets addressed.
- The codebase is full of commented-out code, dead code, and other forms of cruft.

## Symptoms ▲

- [High Technical Debt](high-technical-debt.md)
<br/>  Avoiding refactoring allows technical debt to accumulate unchecked, as structural improvements are never made.
- [Increasing Brittleness](increasing-brittleness.md)
<br/>  Without refactoring, the codebase becomes progressively more fragile as complexity compounds.
- [Code Duplication](code-duplication.md)
<br/>  Developers copy-paste code rather than refactoring shared functionality, leading to widespread duplication.
- [Accumulation of Workarounds](accumulation-of-workarounds.md)
<br/>  Instead of refactoring problematic code, developers build workarounds that add further complexity.
- [Slow Development Velocity](slow-development-velocity.md)
<br/>  Working around structural problems instead of fixing them makes each new change progressively slower to implement.
## Causes ▼

- [Fear of Change](fear-of-change.md)
<br/>  Fear that modifications will break working functionality prevents teams from attempting structural improvements.
- [Fear of Breaking Changes](fear-of-breaking-changes.md)
<br/>  Specific fear of introducing breaking changes in a fragile system makes developers avoid touching existing code.
- [Brittle Codebase](brittle-codebase.md)
<br/>  A fragile codebase makes refactoring genuinely risky, reinforcing avoidance behavior.
- [Test Debt](test-debt.md)
<br/>  Lack of test coverage means there is no safety net to catch regressions during refactoring, making it too risky to attempt.
- [Unrealistic Schedule](unrealistic-schedule.md)
<br/>  Tight deadlines leave no time allocated for refactoring work, prioritizing feature delivery over code improvement.
## Detection Methods ○
- **Code Churn Analysis:** Analyze the history of the codebase to see which files are being modified most frequently. If the same files are being churned over and over again without any improvement in their structure, it is a sign of refactoring avoidance.
- **Technical Debt Backlog:** If the team has a backlog of technical debt items that is constantly growing and never shrinking, it is a clear sign that they are avoiding refactoring.
- **Developer Interviews:** Ask developers about their attitude towards refactoring. If they express fear or reluctance, it is a sign of a problem.
- **Code Quality Metrics:** Track code quality metrics over time. A steady decline in quality is a strong indicator of refactoring avoidance.

## Examples
A team is working on a legacy system that has been in production for over a decade. The code is a mess, but it works. The team is afraid to touch it for fear of breaking it. When they need to add a new feature, they simply copy and paste existing code and modify it slightly. This leads to a lot of code duplication and makes the system even harder to maintain. The team knows that they should refactor the code, but they never do because they are afraid of the consequences. This is a classic example of refactoring avoidance.
