---
title: Direct Feedback
description: Gather feedback from users directly in the software system
category:
- Requirements
- Communication
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/direct-feedback
problems:
- negative-user-feedback
- user-frustration
- customer-dissatisfaction
- no-continuous-feedback-loop
- stakeholder-developer-communication-gap
- feature-gaps
layout: solution
---

## How to Apply ◆

- Add lightweight feedback mechanisms directly into the legacy application: feedback buttons, rating widgets, or contextual surveys on key screens.
- Implement feedback collection that captures the user's current context (page, action, user role) alongside their comments.
- Create a process for triaging and responding to feedback so users see that their input leads to action.
- Analyze feedback patterns to identify the most painful aspects of the legacy system for users.
- Use feedback data to prioritize modernization efforts based on actual user pain points.
- Share aggregated feedback with development teams regularly to maintain empathy with users.

## Tradeoffs ⇄

**Benefits:**
- Provides direct insight into user pain points without relying on intermediaries.
- Enables data-driven prioritization of improvements to the legacy system.
- Builds user trust by demonstrating that their input is valued and acted upon.
- Catches usability issues that internal testing may not reveal.

**Costs:**
- Feedback mechanisms need to be unobtrusive to avoid disrupting the user experience.
- Processing and responding to feedback requires dedicated effort and resources.
- Users may submit feedback about issues outside the development team's control.
- Low response rates may produce a biased sample of user opinions.

## Examples

A legacy enterprise resource planning system receives complaints through a help desk, but by the time feedback reaches developers, it has lost context and urgency. The team adds a small feedback widget to each major screen that lets users rate their experience and add optional comments. Within the first month, they collect hundreds of responses. Analysis reveals that a particular data entry workflow is consistently rated poorly because it requires navigating through seven screens to complete a task that users perform dozens of times daily. This insight, which never surfaced through the help desk channel, becomes the top priority for the next modernization sprint. After streamlining the workflow to three screens, feedback ratings for that area improve dramatically, and users begin proactively suggesting improvements for other areas.
