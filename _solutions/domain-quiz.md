---
title: Domain Quiz
description: Testing domain knowledge through targeted questions
category:
- Communication
- Team
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/domain-quiz
problems:
- knowledge-gaps
- implicit-knowledge
- difficult-developer-onboarding
- incomplete-knowledge
- inconsistent-knowledge-acquisition
layout: solution
---

## How to Apply ◆

- Create quizzes that test developers' understanding of key business concepts, rules, and processes implemented in the legacy system.
- Use quizzes during onboarding to assess new developers' baseline domain knowledge and identify areas where training is needed.
- Include questions about legacy-specific quirks: undocumented business rules, historical decisions, and known edge cases.
- Run periodic domain quizzes with the entire team to surface knowledge gaps before they cause implementation errors.
- Design questions collaboratively with domain experts to ensure they reflect genuinely important business knowledge.
- Use quiz results to guide targeted knowledge-sharing sessions and documentation improvements.

## Tradeoffs ⇄

**Benefits:**
- Reveals knowledge gaps in a low-stakes format before they lead to implementation errors.
- Creates a structured baseline for assessing domain understanding across the team.
- Highlights areas where legacy system documentation is lacking.
- Makes implicit domain knowledge explicit and testable.

**Costs:**
- Quiz creation requires effort from domain experts and experienced developers.
- Quizzes can feel patronizing if not positioned as learning tools rather than evaluations.
- Written quizzes may not capture the nuanced understanding needed for complex domain decisions.
- Maintaining quiz content requires updates as the domain and system evolve.

## Examples

A legacy freight management system has complex rules for calculating shipping rates that depend on carrier contracts, hazmat classifications, and seasonal surcharges. New developers frequently introduce pricing bugs because they do not understand these domain nuances. The team creates a domain quiz covering the twenty most commonly misunderstood business rules, including questions like "What happens to the base rate when a shipment crosses a zone boundary during a seasonal surcharge period?" New developers take the quiz during their second week, and results are discussed in a follow-up session with a senior developer. The quiz reveals that even experienced team members have gaps in their understanding of hazmat classification rules, prompting a focused knowledge-sharing session that prevents a class of bugs that had been recurring quarterly.
