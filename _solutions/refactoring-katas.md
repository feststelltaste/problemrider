---
title: Refactoring Katas
description: Perform regular exercises to improve code quality
category:
- Code
- Team
problems:
- refactoring-avoidance
- fear-of-change
- inexperienced-developers
- skill-development-gaps
- insufficient-design-skills
- lower-code-quality
- procedural-background
- procedural-programming-in-oop-languages
layout: solution
related_solutions:
- slug: incremental-refactoring
  similarity: 0.8
- slug: code-review-process-reform
  similarity: 0.7
- slug: code-reviews
  similarity: 0.7
- slug: code-metrics
  similarity: 0.7
- slug: preparatory-refactoring
  similarity: 0.7
- slug: domain-patterns
  similarity: 0.65
---

## Description

Refactoring katas are short, repeatable coding exercises — such as the Gilded Rose, Tennis Refactoring, or Trip Service kata — that give developers a safe, low-stakes environment to practice specific refactoring techniques like Extract Method or Replace Conditional with Polymorphism before applying them to production code. Because the exercises use small, well-known code samples rather than the team's actual legacy system, developers can experiment freely, make mistakes, and repeat the exercise until a technique becomes automatic, without any risk of breaking something that matters. This distinction matters in legacy contexts because much of the reluctance to refactor old code stems not from a lack of theoretical knowledge but from a lack of practiced, muscle-memory-level skill in applying transformations safely — teams that have spent years only adding features to legacy code, rather than restructuring it, often lack the confidence to touch it at all. Regular kata practice rebuilds that confidence gradually and creates a shared vocabulary of technique names and approaches across the team, so that when someone proposes extracting a value object or replacing a conditional with polymorphism during a real code review, everyone understands what is meant and how to do it safely. Over time, the skills exercised in katas transfer directly to the legacy codebase, turning refactoring from an intimidating, occasional event into a routine part of daily work.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Schedule regular practice sessions (e.g., weekly one-hour slots) where team members work through refactoring exercises
- Use well-known katas such as the Gilded Rose, Tennis Refactoring, or Trip Service kata that simulate legacy code scenarios
- Practice in pairs or small groups to share techniques and build a common refactoring vocabulary
- Focus each session on a specific technique: Extract Method, Replace Conditional with Polymorphism, Introduce Parameter Object
- Apply techniques learned in katas to actual legacy code in low-risk areas to reinforce skills
- Track which refactoring patterns the team is most and least comfortable with to guide future sessions
- Use the IDE's automated refactoring tools during katas to build muscle memory for safe transformations

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Builds team confidence to refactor production legacy code safely
- Creates shared refactoring vocabulary and approach across the team
- Reduces fear of changing legacy code by providing hands-on practice in a safe environment
- Improves code quality incrementally as developers apply learned techniques daily

**Costs and Risks:**
- Requires dedicated time that competes with feature delivery pressure
- Benefits are gradual and hard to measure in the short term
- May feel academic if katas are not connected to real codebase challenges
- Risk of diminishing returns if the same exercises are repeated without progression

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A development team maintaining a legacy ERP system was reluctant to refactor a 3,000-line order processing class because no one felt confident making structural changes to such critical code. The tech lead introduced bi-weekly refactoring kata sessions, starting with the Gilded Rose kata to practice extracting methods and introducing abstractions. After two months, the team applied the Extract Class refactoring to the order processing class during a planned sprint, breaking it into five focused classes. The kata practice had given them both the skills and the confidence to perform the refactoring safely, and the resulting code was significantly easier to maintain.
