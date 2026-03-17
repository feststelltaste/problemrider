---
title: Assumption-Based Development
description: Developers make decisions based on assumptions about requirements or
  user needs rather than validating their understanding.
category:
- Communication
- Process
- Requirements
related_problems:
- slug: convenience-driven-development
  similarity: 0.55
- slug: inexperienced-developers
  similarity: 0.5
- slug: decision-avoidance
  similarity: 0.5
- slug: feedback-isolation
  similarity: 0.5
- slug: decision-paralysis
  similarity: 0.5
- slug: inadequate-requirements-gathering
  similarity: 0.5
layout: problem
---

## Description

Assumption-based development occurs when developers make decisions about functionality, user interface design, business logic, or technical approach based on their assumptions about what users need or what stakeholders want, rather than validating these assumptions through direct communication or research. While some assumptions are inevitable in development, excessive reliance on assumptions leads to solutions that don't match actual needs and require costly rework.

## Indicators ⟡

- Developers proceed with implementation without asking clarifying questions
- Design decisions are made without consulting users or stakeholders
- Business logic is implemented based on developer interpretation of requirements
- Edge cases and error conditions are handled based on developer assumptions
- User interface designs are created without user input or testing

## Symptoms ▲

- [Implementation Rework](implementation-rework.md)
<br/>  When assumptions about requirements prove wrong, completed work must be reworked to match actual needs.
- [Misaligned Deliverables](misaligned-deliverables.md)
<br/>  Software built on unvalidated assumptions delivers features that don't match what stakeholders actually need.
- [Wasted Development Effort](wasted-development-effort.md)
<br/>  Development effort spent building features based on incorrect assumptions is effectively wasted.
- [Negative User Feedback](negative-user-feedback.md)
<br/>  Users provide negative feedback when delivered features don't match their actual needs and workflows.
- [Feature Gaps](feature-gaps.md)
<br/>  Assumptions about user needs lead to building the wrong features while actual needs remain unaddressed.
- [Second-System Effect](second-system-effect.md)
<br/>  The second-system effect leads to an overengineered replacement.

## Causes ▼

- [Inadequate Requirements Gathering](inadequate-requirements-gathering.md)
<br/>  When requirements are not properly gathered, developers fill in gaps with their own assumptions.
- [Stakeholder-Developer Communication Gap](stakeholder-developer-communication-gap.md)
<br/>  Poor communication channels between stakeholders and developers force developers to guess at requirements.
- [Deadline Pressure](deadline-pressure.md)
<br/>  Time pressure causes developers to proceed with assumptions rather than taking time to validate requirements.
- [Feedback Isolation](feedback-isolation.md)
<br/>  When developers are isolated from user feedback, they must rely on assumptions about user needs.
## Detection Methods ○

- **Assumption Documentation:** Track what assumptions developers are making during development
- **Stakeholder Validation Frequency:** Monitor how often developers check assumptions with stakeholders
- **Rework Analysis:** Analyze how much rework is caused by incorrect assumptions
- **User Feedback Correlation:** Compare user feedback with original developer assumptions
- **Requirements Clarity Assessment:** Evaluate how clear and specific requirements are before development begins

## Examples

A development team builds a customer search feature assuming that users will search primarily by company name, so they optimize the search interface and algorithms for company name searches. When they deploy the feature, they discover that users actually search most frequently by contact person names and email addresses, making the search interface frustrating to use and the search results poor quality. The search feature must be redesigned and rebuilt to support the actual search patterns. Another example involves developers building a reporting system who assume that users want to see data in real-time, so they implement complex real-time data processing. However, users actually prefer to work with stable daily snapshots of data for consistency in their analysis, and the real-time updates make it difficult to create reproducible reports. The real-time complexity adds no value and creates maintenance overhead for functionality users don't want.
