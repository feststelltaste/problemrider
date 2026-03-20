---
title: Plain Language
description: Use simple and clear formulations
category:
- Communication
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/plain-language/
problems:
- user-confusion
- poor-user-experience-ux-design
- poor-documentation
- user-frustration
- difficult-developer-onboarding
- negative-user-feedback
- knowledge-gaps
layout: solution
---

## How to Apply ◆

> Legacy systems accumulate technical jargon, cryptic abbreviations, and developer-oriented language in their interfaces and documentation. Plain language replaces this with clear, user-oriented writing.

- Audit all user-facing text in the legacy system, including labels, buttons, menu items, error messages, help text, and tooltips. Flag instances of technical jargon, abbreviations, and overly complex sentences.
- Rewrite error messages to explain what happened, why it happened, and what the user can do about it. Replace messages like "ERR_4021: Transaction rollback" with "Your changes could not be saved because another user updated this record. Please refresh and try again."
- Use action-oriented button labels that describe what will happen, such as "Save and continue" or "Delete this record," instead of generic labels like "OK," "Submit," or "Execute."
- Write in short sentences using active voice. Avoid passive constructions like "The record has been updated" when "We updated the record" or "Record updated" is clearer.
- Define a writing style guide for all user-facing text that covers tone, terminology, capitalization, and punctuation. Share it with all developers who write interface text.
- Test new text with representative users to verify comprehension, especially for critical workflows where misunderstanding has consequences.

## Tradeoffs ⇄

> Plain language makes the system more accessible and reduces errors, but requires writing skill and ongoing attention to language quality.

**Benefits:**

- Reduces user confusion by making interface text immediately understandable without domain expertise or system experience.
- Decreases support requests caused by users who do not understand what a button does or what an error message means.
- Shortens onboarding time because new users can understand the interface without memorizing jargon or abbreviations.
- Makes the system accessible to a broader audience, including users with lower literacy levels or non-native speakers.

**Costs and Risks:**

- Rewriting all user-facing text in a large legacy system is a significant effort that must be prioritized alongside other improvements.
- Achieving consensus on plain language alternatives for established technical terms can be difficult when different groups have strong preferences.
- Oversimplifying technical concepts in an expert-facing system can feel patronizing to experienced users who prefer precise terminology.
- Maintaining plain language standards requires vigilance during code reviews, as developers tend to revert to technical language without guidance.

## Examples

> Legacy systems often feel hostile to users not because of poor functionality but because of impenetrable language.

A legacy legal case management system used by court clerks displays error messages that were written by developers and never reviewed for clarity. Messages like "Constraint violation: FK_CASE_PARTY_REF integrity check failed" appear when a clerk tries to delete a party record that is still linked to an active case. Clerks have no idea what the message means and either call IT support or try random things until the error goes away. The team rewrites all two hundred error messages in plain language, with the constraint violation message becoming "This party cannot be removed because they are linked to one or more active cases. To remove them, first reassign or close the linked cases." Each message now includes a specific action the user can take. Support calls related to error messages drop by over half, and clerks report feeling more confident using the system because they can understand and respond to problems independently.
