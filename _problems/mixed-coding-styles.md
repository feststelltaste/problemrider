---
title: Mixed Coding Styles
description: A situation where different parts of the codebase use different formatting,
  naming conventions, and design patterns.
category:
- Code
related_problems:
- slug: inconsistent-coding-standards
  similarity: 0.75
- slug: inconsistent-codebase
  similarity: 0.75
- slug: inconsistent-naming-conventions
  similarity: 0.65
- slug: style-arguments-in-code-reviews
  similarity: 0.65
- slug: spaghetti-code
  similarity: 0.65
- slug: undefined-code-style-guidelines
  similarity: 0.65
layout: problem
---

## Description
Mixed coding styles is a situation where different parts of the codebase use different formatting, naming conventions, and design patterns. This is a common problem in long-running projects, especially those that have been worked on by many different people over the years. Mixed coding styles can lead to a number of problems, including a decrease in readability, an increase in cognitive load, and a general slowdown in development velocity.

## Indicators ⟡
- The codebase is difficult to read and understand.
- There are multiple ways to do the same thing.
- The codebase is a mixture of different styles and conventions.
- There is no style guide for the project, or it exists but is not enforced.

## Symptoms ▲

- [Difficult Code Comprehension](difficult-code-comprehension.md)
<br/>  Inconsistent formatting, naming, and patterns force developers to mentally context-switch, making code harder to read and understand.
- [Style Arguments in Code Reviews](style-arguments-in-code-reviews.md)
<br/>  Without a consistent style, code reviews devolve into debates about formatting and naming preferences rather than logic and design.
- [Slow Feature Development](slow-feature-development.md)
<br/>  Developers spend extra time deciphering inconsistent code patterns, slowing down the pace of feature delivery.
- [New Hire Frustration](new-hire-frustration.md)
<br/>  New developers joining the project are confused by inconsistent conventions and struggle to know which style to follow.
- [Inconsistent Codebase](inconsistent-codebase.md)
<br/>  Mixed coding styles directly contribute to an overall inconsistent codebase that lacks coherence across modules.

## Causes ▼
- [Undefined Code Style Guidelines](undefined-code-style-guidelines.md)
<br/>  Without defined and enforced style guidelines, each developer applies their own preferred conventions.
- [High Turnover](high-turnover.md)
<br/>  Frequent developer turnover brings in new people with different coding habits, each leaving their stylistic imprint on the codebase.
- [Inconsistent Coding Standards](inconsistent-coding-standards.md)
<br/>  Lack of agreed-upon or enforced coding standards directly leads to different parts of the codebase using different styles.
- [Insufficient Code Review](insufficient-code-review.md)
<br/>  Without thorough code reviews that enforce consistency, style deviations accumulate unchecked over time.
- [Procedural Programming in OOP Languages](procedural-programming-in-oop-languages.md)
<br/>  Procedural code mixed with OOP code from other developers creates inconsistent coding patterns across the codebase.

## Detection Methods ○
- **Manual Code Inspection:** The inconsistency is often obvious from simply browsing the codebase.
- **Run a Linter or Formatter:** Run a tool like ESLint, Prettier, RuboCop, or Black on the codebase and observe the large number of reported violations.
- **Team Surveys:** Ask developers if they find the codebase easy to read and understand.
- **Analyze Code Review Comments:** Look for a high frequency of comments related to style and formatting.

## Examples
A large enterprise application has been developed by multiple teams over a decade. One module uses camelCase for variables, another uses snake_case, and a third mixes both. Indentation varies between tabs and spaces, and brace styles are inconsistent. This makes it very difficult for any single developer to work across modules efficiently. A new developer joins and submits a pull request that is rejected multiple times due to style violations that were never explicitly communicated, leading to frustration and delays.
