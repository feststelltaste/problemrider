---
title: Increased Customer Support Load
description: Users contact support more frequently due to frustration or inability
  to complete tasks.
category:
- Business
- Code
- Process
related_problems:
- slug: customer-dissatisfaction
  similarity: 0.6
- slug: user-frustration
  similarity: 0.55
- slug: increased-bug-count
  similarity: 0.55
- slug: user-confusion
  similarity: 0.55
- slug: increased-cognitive-load
  similarity: 0.55
- slug: maintenance-cost-increase
  similarity: 0.5
solutions:
- contextual-help
- frequently-asked-questions-faq
- input-constraints-and-defaults
- confirmation-dialogs
- auto-save
- product-strategy-alignment
layout: problem
---

## Description

Increased customer support load occurs when technical problems, poor user experience design, or system defects cause users to contact support more frequently than necessary. This creates a cascading effect where technical problems not only affect users directly but also strain support resources, increase operational costs, and divert attention from other business activities. The support team becomes overwhelmed with issues that should have been prevented through better software quality.

## Indicators ⟡
- Support ticket volume increases without corresponding user growth
- High percentage of support requests relate to software bugs or usability issues
- Support representatives spend significant time on recurring technical problems
- Customer satisfaction scores decline due to support experience
- Support costs increase disproportionately to user base growth

## Symptoms ▲

- [Maintenance Cost Increase](maintenance-cost-increase.md)
<br/>  More support tickets require more support staff and operational resources, increasing overall maintenance costs.
- [Operational Overhead](operational-overhead.md)
<br/>  Support staff spends time on technical issues that should have been prevented, diverting resources from planned activities.
## Causes ▼

- [Increased Bug Count](increased-bug-count.md)
<br/>  More production bugs mean more users encounter problems and need to contact support.
- [Poor User Experience (UX) Design](poor-user-experience-ux-design.md)
<br/>  Confusing or difficult interfaces cause users to seek help for tasks they should be able to complete independently.
- [Inadequate Error Handling](inadequate-error-handling.md)
<br/>  Unclear or missing error messages leave users unable to resolve issues on their own, forcing them to contact support.
## Detection Methods ○
- **Support Ticket Categorization:** Classify tickets to identify what percentage relate to technical issues vs. legitimate support needs
- **Ticket Volume Trends:** Monitor support request volume relative to user base growth
- **Common Issue Analysis:** Track the most frequent support requests to identify recurring technical problems
- **Support Cost Analysis:** Calculate the cost impact of technical issues on support operations
- **User Self-Service Success Rate:** Measure how often users can resolve issues without contacting support

## Examples

An online banking application has a confusing password reset process that frequently fails due to technical issues with email delivery and session management. Instead of the intended 2-step self-service process, users often need to call support to reset their passwords. What should be a simple automated workflow generates hundreds of support calls per week, requiring support representatives to manually reset accounts. The support team spends 40% of their time on password-related issues that should be resolved automatically. This not only increases support costs but also creates security risks as support staff must verify user identity over the phone for account access. Another example involves an e-commerce platform where the checkout process frequently fails with unclear error messages. Instead of showing specific information like "credit card declined" or "shipping address invalid," the system displays generic "transaction failed" messages. Users contact support to complete their purchases, turning what should be automated sales into manual processes that require human intervention. The support team becomes a bottleneck for revenue generation, and many users abandon their purchases rather than deal with the support process.
