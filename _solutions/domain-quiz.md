---
title: Domain Quiz
description: Testing domain knowledge through targeted questions
category:
- Communication
- Team
problems:
- knowledge-gaps
- implicit-knowledge
- difficult-developer-onboarding
- incomplete-knowledge
- inconsistent-knowledge-acquisition
layout: solution
related_solutions:
- slug: knowledge-sharing-practices
  similarity: 0.7
- slug: knowledge-base
  similarity: 0.7
- slug: domain-experts
  similarity: 0.7
- slug: pair-and-mob-programming
  similarity: 0.7
- slug: subject-matter-reviews
  similarity: 0.7
- slug: structured-onboarding-program
  similarity: 0.65
---

## Description

A domain quiz is a structured set of targeted questions, designed collaboratively with domain experts, that tests developers' understanding of the business concepts, rules, and edge cases embedded in a system — used during onboarding or run periodically with the whole team to surface gaps in domain knowledge before those gaps cause implementation errors. This is a deliberately low-stakes format for making implicit knowledge explicit and testable: legacy systems typically encode years of accumulated business rules and quirks that were never fully documented, so a developer's confidence in their own understanding is not a reliable signal of whether that understanding is actually correct or complete. Running a quiz surfaces exactly where that confidence is misplaced — often revealing that even experienced team members carry gaps around specific undocumented rules or historical decisions — before those gaps manifest as pricing errors, incorrect calculations, or other business-logic defects in production. Because the questions are written jointly with domain experts, the quiz results also function as a diagnostic for the documentation itself, pointing directly at which areas of the system's actual behavior are least well captured anywhere written down. Used well, a domain quiz functions as a recurring pulse check on institutional knowledge rather than a one-time onboarding gate, which matters most in systems where that knowledge is thin, uneven across the team, and at risk of walking out the door with any one person.

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

## How It Could Be

A legacy freight management system has complex rules for calculating shipping rates that depend on carrier contracts, hazmat classifications, and seasonal surcharges. New developers frequently introduce pricing bugs because they do not understand these domain nuances. The team creates a domain quiz covering the twenty most commonly misunderstood business rules, including questions like "What happens to the base rate when a shipment crosses a zone boundary during a seasonal surcharge period?" New developers take the quiz during their second week, and results are discussed in a follow-up session with a senior developer. The quiz reveals that even experienced team members have gaps in their understanding of hazmat classification rules, prompting a focused knowledge-sharing session that prevents a class of bugs that had been recurring quarterly.
