---
title: Monolithic Functions and Classes
description: Individual functions or classes perform too many unrelated responsibilities,
  making them difficult to understand and modify.
category:
- Architecture
- Code
related_problems:
- slug: god-object-anti-pattern
  similarity: 0.75
- slug: poorly-defined-responsibilities
  similarity: 0.7
- slug: monolithic-architecture-constraints
  similarity: 0.7
- slug: excessive-class-size
  similarity: 0.65
- slug: over-reliance-on-utility-classes
  similarity: 0.6
- slug: bloated-class
  similarity: 0.6
layout: problem
---

## Description

Monolithic functions and classes are code components that have grown to handle multiple, often unrelated responsibilities within a single unit. These "god functions" or "god classes" violate the Single Responsibility Principle and become central points of complexity that are difficult to understand, modify, test, or reuse. They often emerge organically as features are added over time, with developers continuously extending existing functions rather than creating new, focused components.

## Indicators ⟡
- Functions that are hundreds or thousands of lines long
- Classes with dozens of methods and instance variables
- Functions or methods that require extensive scrolling to review completely
- Code that handles multiple distinct business concepts or technical concerns
- Difficulty summarizing what a function or class does in a single sentence

## Symptoms ▲

- [Difficult Code Comprehension](difficult-code-comprehension.md)
<br/>  Functions and classes handling many responsibilities are extremely difficult to understand, as developers must grasp all concerns simultaneously.
- [Merge Conflicts](merge-conflicts.md)
<br/>  Large functions and classes modified by multiple developers for different features frequently result in merge conflicts.
- [Difficult Code Reuse](difficult-code-reuse.md)
<br/>  When functionality is bundled into monolithic units, extracting and reusing individual pieces becomes impractical.
- [High Bug Introduction Rate](high-bug-introduction-rate.md)
<br/>  Complex, multi-responsibility functions are prone to bugs because changes to one concern can inadvertently affect others.
- [Difficult to Test Code](difficult-to-test-code.md)
<br/>  Testing monolithic functions requires extensive setup and mocking of many dependencies, making thorough testing impractical.

## Causes ▼

- [Poorly Defined Responsibilities](poorly-defined-responsibilities.md)
<br/>  Without clear responsibility assignment, developers keep adding functionality to existing components rather than creating focused new ones.
- [Insufficient Design Skills](insufficient-design-skills.md)
<br/>  Developers lacking design skills fail to recognize when a function or class should be decomposed into smaller, focused units.
- [Short-Term Focus](short-term-focus.md)
<br/>  Under pressure to deliver quickly, developers extend existing functions rather than investing time in proper decomposition.
- [Fear of Change](fear-of-change.md)
<br/>  Developers avoid breaking up large functions due to the risk of introducing bugs in already working code, allowing them to grow further.
## Detection Methods ○
- **Code Metrics Tools:** Use static analysis tools to measure function length, cyclomatic complexity, and class size
- **Responsibility Analysis:** Identify functions or classes that handle multiple distinct business or technical concerns
- **Code Review Patterns:** Look for reviews that mention difficulty understanding or testing specific components
- **Change Frequency Analysis:** Components that are modified frequently may be handling too many responsibilities
- **Testing Complexity:** Identify components that require extensive setup or multiple test scenarios

## Examples

An e-commerce application has a single `processOrder` function that handles payment processing, inventory updates, customer notifications, order logging, tax calculations, shipping arrangements, loyalty point updates, and fraud detection. This 800-line function is modified whenever any aspect of order processing changes, making it a constant source of bugs and merge conflicts. Testing this function requires mocking payment systems, databases, email services, and multiple external APIs. When a simple change is needed to the tax calculation logic, developers must understand the entire order processing workflow and risk breaking payment processing or inventory management. Another example involves a `UserManager` class with 45 methods that handles user authentication, profile management, permissions, password reset, email verification, activity logging, and social media integration. Any change to user functionality requires understanding this massive class, and testing individual features like password reset requires initializing the entire user management system.
