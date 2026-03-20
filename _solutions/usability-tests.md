---
title: Usability Tests
description: Conducting tests with representative users
category:
- Testing
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/usability-tests/
problems:
- poor-user-experience-ux-design
- user-frustration
- user-confusion
- negative-user-feedback
- customer-dissatisfaction
- feature-gaps
- shadow-systems
- stakeholder-developer-communication-gap
layout: solution
---

## How to Apply ◆

> Legacy systems are rarely tested with actual users, so usability problems accumulate undetected for years. Systematic usability testing reveals problems that developers and product owners cannot see because they are too familiar with the system.

- Recruit five to eight representative users per test session. Research shows that five users uncover approximately eighty percent of usability issues. Select participants who represent the actual user base in terms of role, experience level, and technical comfort.
- Design task-based test scenarios that reflect real workflows rather than artificial exercises. Ask users to complete tasks they would normally perform, such as filing a report, processing an order, or looking up a customer record.
- Use the think-aloud protocol where participants verbalize their thought process as they work. This reveals not just what users do but why they do it and where they become confused.
- Record sessions with screen capture and audio so the team can review observations after the test. Direct observation during the session captures immediate reactions, but recordings reveal details that are easy to miss live.
- Analyze results by severity and frequency. Prioritize issues that cause task failure or significant delay over cosmetic or preferential issues.
- Conduct usability tests at regular intervals, not just once. Run tests before and after major interface changes to measure improvement and catch regressions.

## Tradeoffs ⇄

> Usability testing provides the most direct evidence of user experience problems, but requires time, planning, and access to representative users.

**Benefits:**

- Reveals usability problems that are invisible to the development team because they have adapted to the system's quirks over time.
- Provides concrete, observable evidence for prioritizing UX improvements, making it easier to justify investment in usability work.
- Identifies shadow systems and workarounds that users have developed, exposing hidden requirements and feature gaps.
- Validates that proposed improvements actually help users rather than introducing new problems, preventing wasted development effort.

**Costs and Risks:**

- Recruiting representative users takes time, and users pulled from their regular work may resent the interruption if the process is not well organized.
- Usability tests produce qualitative data that requires skilled interpretation. Inexperienced observers may focus on subjective preferences rather than genuine usability issues.
- Testing a legacy system with severe usability problems can produce an overwhelming number of findings, requiring discipline to prioritize rather than trying to fix everything at once.
- There is a risk of over-indexing on the behavior of a small number of test participants, making it important to distinguish between individual preferences and genuine design problems.

## Examples

> Organizations that have never conducted usability tests on their legacy systems are consistently surprised by what they discover.

A legacy human resources system has been in use for eight years, and the development team believes the interface is adequate because support ticket volume is manageable. A usability test with six HR generalists reveals a different picture: every participant struggles with the same three workflow steps, all six develop different workarounds for the same navigation problem, and two participants fail to complete a routine task within the time limit because they cannot find the correct screen. The test also reveals that users have created an unofficial wiki with annotated screenshots explaining how to perform common tasks, a shadow documentation system the development team did not know existed. The test results provide specific, prioritized improvement targets that the team addresses over two sprints. A follow-up usability test three months later shows measurable improvement in task completion time and a reduction in errors for the three problem workflows.
