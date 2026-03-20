---
title: User-Centered Design
description: Incorporate users' needs, expectations, and abilities from the beginning
category:
- Requirements
- Business
quality_tactics_url: https://qualitytactics.de/en/usability/user-centered-design/
problems:
- poor-user-experience-ux-design
- user-frustration
- user-confusion
- user-trust-erosion
- negative-user-feedback
- customer-dissatisfaction
- negative-brand-perception
- feature-gaps
- shadow-systems
- competitive-disadvantage
- high-client-side-resource-consumption
- high-resource-utilization-on-client
- inefficient-frontend-code
layout: solution
---

## How to Apply ◆

> Legacy systems frequently suffer from interfaces that were designed around technical constraints and developer assumptions rather than actual user needs. Introducing user-centered design practices into legacy modernization efforts ensures that improvements address real problems rather than imagined ones.

- Conduct structured user research before redesigning any legacy interface. Interview at least five representative users per role to understand their actual workflows, pain points, and workarounds. Legacy systems often have users who have adapted to the system's quirks over years, and their insights reveal both what must be preserved and what must change.
- Create user journey maps that document how users currently accomplish their goals with the legacy system, including the manual workarounds, shadow spreadsheets, and external tools they use to compensate for system deficiencies. These maps expose the true scope of UX problems that the development team may not even be aware of.
- Establish a regular usability testing cadence where real users attempt to complete representative tasks using the system. For legacy systems, this is often revealing because developers who have grown accustomed to the interface cannot see the confusion that new or occasional users experience.
- Apply progressive disclosure principles when modernizing complex legacy interfaces: show users only the controls and information relevant to their current task, and provide access to advanced features through deliberate exploration rather than overwhelming the primary interface.
- Implement an in-application feedback mechanism that allows users to report issues and frustrations directly in context, capturing the specific screen, workflow step, and action that caused the problem. This provides continuous, low-friction feedback that is far more actionable than periodic surveys.
- Maintain a design system or style guide that ensures consistency across all parts of the application, including both modernized and not-yet-modernized sections. Inconsistency between different parts of the system is a major source of user confusion in legacy applications undergoing incremental improvement.
- Prioritize UX improvements in the areas of the application with the highest user traffic and the most support tickets, rather than attempting a complete redesign. Data-driven prioritization ensures that limited design resources address the problems that affect the most users.
- Include accessibility standards (WCAG 2.1 AA minimum) in every UX improvement, not as an afterthought but as a core design constraint. Legacy systems frequently lack accessibility features, and retrofitting them during modernization is far cheaper than addressing them separately.

## Tradeoffs ⇄

> User-centered design transforms legacy systems from inside-out (built for developers) to outside-in (built for users), but requires sustained investment in research and design capabilities that many legacy-focused organizations lack.

**Benefits:**

- Directly addresses the root cause of user frustration, negative feedback, and customer dissatisfaction by designing interfaces based on validated user needs rather than developer assumptions.
- Reduces shadow system proliferation by making official systems genuinely useful for their intended users, eliminating the motivation to build workarounds.
- Provides measurable business impact through improved task completion rates, reduced support ticket volumes, and increased user engagement — metrics that justify continued investment in UX improvements.
- Builds user trust incrementally by demonstrating that the organization listens to and acts on user feedback, counteracting the erosion of confidence caused by years of poor experiences.
- Prevents feature gaps by validating functionality requirements with actual users before development, ensuring that what gets built matches what users actually need.

**Costs and Risks:**

- User research and usability testing require dedicated time and resources that compete with feature delivery, and organizations accustomed to shipping without user input may resist the perceived slowdown.
- Legacy system users who have adapted to existing workflows over years may initially resist interface changes, even when the new design is objectively better, requiring careful change management.
- Incremental UX improvements in a legacy system can create temporary inconsistencies between modernized and unmodernized sections, potentially increasing user confusion in the short term.
- Investing heavily in UX design for a system that may be replaced creates a tension between improving the current experience and planning for a future platform.
- Organizations without design expertise must either hire UX professionals or train existing staff, both of which require investment before benefits materialize.

## Examples

> The following scenarios illustrate how user-centered design practices address the specific UX challenges found in legacy systems.

A hospital information system built over fifteen years has a patient scheduling interface that requires nurses to navigate through seven screens to complete a single booking. The interface was designed by developers to mirror the database structure rather than the clinical workflow. After conducting shadowing sessions with nursing staff, the UX team discovers that nurses have developed elaborate paper-based checklists to remember the sequence of screens and fields. The team redesigns the scheduling flow into a single-page wizard that follows the actual clinical workflow, with progressive disclosure for exceptional cases. Support tickets related to scheduling errors drop by 65% within three months, and the nursing staff who previously maintained shadow spreadsheets to track bookings return to using the official system.

An insurance company's claims processing system generates persistent negative user feedback due to inconsistent terminology across different sections — the same concept is labeled "claim number," "case ID," and "reference code" depending on which module the user is in. A design audit reveals 47 instances of inconsistent terminology. The team establishes a design system with a standardized glossary, applies it systematically during routine maintenance work, and implements contextual help tooltips that explain terms users find confusing. Over six months, the call volume to the internal help desk drops by 40%, and new employee onboarding time for the claims system decreases from three weeks to one week. The consistent vocabulary also reduces data entry errors where agents typed information into wrong fields because they misunderstood the labels.
