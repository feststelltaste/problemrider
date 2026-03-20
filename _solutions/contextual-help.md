---
title: Contextual Help
description: Providing help information and explanations directly in the context of the current task
category:
- Communication
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/contextual-help/
problems:
- user-confusion
- user-frustration
- poor-user-experience-ux-design
- poor-documentation
- difficult-developer-onboarding
- increased-customer-support-load
- negative-user-feedback
- knowledge-gaps
layout: solution
---

## How to Apply ◆

> Legacy systems often have either no help at all or a separate help manual that users never consult. Contextual help embeds guidance directly where users need it, reducing confusion and support requests.

- Add tooltip explanations to form fields, buttons, and interface elements that cause frequent confusion or support requests. Use data from support tickets to identify the highest-priority targets.
- Implement inline help text for complex fields that explains what the field is for, what format is expected, and what the consequences of different values are. Legacy systems often have fields whose purpose is clear only to the original developer.
- Create context-sensitive help panels that display relevant guidance based on the current screen and the user's current task, rather than requiring users to search through a separate help system.
- Add explanatory text to error messages that tells users not just what went wrong but what to do about it. Legacy error messages often display cryptic codes or technical jargon.
- Use progressive disclosure for help content: show brief hints by default and provide access to detailed explanations for users who need more information.
- Keep help content close to the element it describes. Users should not need to leave their current context to find an explanation.

## Tradeoffs ⇄

> Contextual help provides immediate answers at the point of need, but requires ongoing maintenance as the system evolves.

**Benefits:**

- Reduces support ticket volume by answering common questions directly within the interface, before users reach out for help.
- Decreases onboarding time because new users can learn the system while using it rather than studying a separate manual.
- Addresses knowledge gaps caused by poor or outdated documentation by placing current, accurate guidance where it matters most.
- Builds user confidence by providing reassurance at decision points, reducing hesitation and errors.

**Costs and Risks:**

- Help content must be maintained alongside the application. Outdated contextual help that describes behavior that has changed is worse than no help at all.
- Excessive tooltips and inline help can clutter the interface and annoy experienced users who do not need guidance, requiring careful balance.
- Writing effective help content requires understanding user tasks at a detailed level, which may require collaboration with domain experts.
- Translating contextual help into multiple languages adds to the localization burden in internationally deployed legacy systems.

## How It Could Be

> Legacy systems often suffer from a documentation gap where the only people who understand the interface are those who built it years ago.

A legacy accounting system has a "Period Close" process that involves setting flags across multiple screens in a specific order. The process is documented in a forty-page procedural manual that accounting staff print and follow step by step each month. When the team adds contextual help panels to each screen in the period close workflow, showing what this specific step accomplishes and what comes next, the accounting staff can perform the process without consulting the manual. New accountants who previously needed a senior colleague to walk them through their first several month-end closes can now complete the process independently using the embedded guidance. The support team reports that period-close-related questions, which previously spiked every month-end, are nearly eliminated.
