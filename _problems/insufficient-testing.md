---
title: Quality Blind Spots
description: The testing process is not comprehensive enough, leading to a high defect
  rate in production.
category:
- Code
- Process
related_problems:
- slug: high-defect-rate-in-production
  similarity: 0.75
- slug: quality-blind-spots
  similarity: 0.7
- slug: inadequate-test-data-management
  similarity: 0.65
- slug: testing-complexity
  similarity: 0.65
- slug: insufficient-design-skills
  similarity: 0.65
- slug: complex-deployment-process
  similarity: 0.6
layout: problem
---

## Description
Insufficient testing is a major cause of poor software quality. When a product is not thoroughly tested, it is likely to have a high number of bugs, which can lead to a poor user experience, a loss of trust, and a significant amount of rework. A comprehensive testing strategy should include a mix of automated and manual testing, and it should be integrated into the development process from the very beginning. Investing in testing is an investment in the quality and stability of the product.

## Indicators ⟡
- The team has no automated tests.
- The team has a low level of test coverage.
- The team is constantly finding bugs in production.
- The team is afraid to make changes to the codebase for fear of breaking something.

## Symptoms ▲

- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  Without comprehensive testing, more defects escape to production where they affect users.
- [Regression Bugs](regression-bugs.md)
<br/>  Insufficient test coverage means changes frequently break existing functionality without detection before release.
- [Fear of Change](fear-of-change.md)
<br/>  Developers become afraid to modify code when there are no tests to verify their changes do not break anything.
- [Frequent Hotfixes and Rollbacks](frequent-hotfixes-and-rollbacks.md)
<br/>  Production defects from inadequate testing require emergency fixes and rollbacks.
- [Constant Firefighting](constant-firefighting.md)
<br/>  Teams spend significant time dealing with production issues that testing should have caught.
- [Customer Dissatisfaction](customer-dissatisfaction.md)
<br/>  Frequent production bugs caused by insufficient testing directly impact user experience and satisfaction.

## Causes ▼
- [Deadline Pressure](deadline-pressure.md)
<br/>  Under time pressure, testing is often the first activity to be cut or reduced.
- [Short-Term Focus](short-term-focus.md)
<br/>  Prioritizing immediate feature delivery over long-term quality leads to underinvestment in testing.
- [Insufficient Design Skills](insufficient-design-skills.md)
<br/>  Poorly designed code is difficult to test, which discourages comprehensive testing efforts.
- [Inadequate Test Infrastructure](inadequate-test-infrastructure.md)
<br/>  Lack of proper test environments and tooling makes comprehensive testing impractical.
- [Testing Complexity](testing-complexity.md)
<br/>  The high effort required to test duplicated functionality leads to insufficient test coverage overall.

## Detection Methods ○

- **Bug Tracking Metrics:** Monitor the number of bugs found in production versus pre-production environments.
- **Code Coverage Tools:** Use tools to measure the percentage of code executed by tests.
- **Test Automation Reports:** Analyze reports from automated test runs to identify gaps or failures.
- **Retrospectives:** Discuss testing effectiveness and identify areas for improvement in team retrospectives.
- **Manual Test Case Review:** Review manual test cases to identify areas where automation could be introduced or coverage improved.

## Examples
A new feature is released, and immediately, users report that a critical workflow is broken. Investigation reveals that while individual components were tested, the end-to-end flow involving multiple services was never tested in an integrated environment. In another case, a developer makes a small change to a utility function. Without unit tests for that function, they don't realize it has a side effect that breaks another, seemingly unrelated part of the application, leading to a regression bug in production. This problem often stems from a culture that prioritizes speed over quality, or a lack of understanding of the long-term benefits of a robust testing strategy. It can lead to significant technical debt and a constant state of firefighting.
