---
title: Insufficient Design Skills
description: The development team lacks the necessary skills and experience to design
  and build well-structured, maintainable software.
category:
- Code
- Team
related_problems:
- slug: inexperienced-developers
  similarity: 0.7
- slug: misunderstanding-of-oop
  similarity: 0.7
- slug: inconsistent-codebase
  similarity: 0.7
- slug: incomplete-knowledge
  similarity: 0.65
- slug: difficult-code-reuse
  similarity: 0.65
- slug: insufficient-testing
  similarity: 0.65
solutions:
- architecture-reviews
- boring-technologies
- solid-principles
- technical-skills-development
- pattern-language
- refactoring-katas
layout: problem
---

## Description
Insufficient design skills are a major contributor to the creation of legacy code. When a development team lacks the necessary skills and experience to design and build well-structured, maintainable software, they are likely to create a system that is difficult to understand, modify, and test. This can lead to a number of problems, including a high rate of bugs, a slow development velocity, and a great deal of frustration for the development team. Insufficient design skills are a common problem in the software industry, and it can be difficult to address.

## Indicators ⟡
- The codebase is a "big ball of mud."
- The team is constantly struggling with technical debt.
- The team is not able to deliver new features in a timely manner.
- The team is not proud of the code that they are writing.

## Symptoms ▲

- [Spaghetti Code](spaghetti-code.md)
<br/>  Lack of design skills leads to poorly structured code with tangled control flow and unclear organization.
- [High Coupling and Low Cohesion](high-coupling-low-cohesion.md)
<br/>  Developers without design skills create tightly coupled modules with poor separation of concerns.
- [High Technical Debt](high-technical-debt.md)
<br/>  Poor design decisions accumulate technical debt that becomes increasingly expensive to address.
- [Difficult Code Reuse](difficult-code-reuse.md)
<br/>  Poorly designed components are hard to reuse because they lack clear interfaces and proper abstractions.
- [Slow Development Velocity](slow-development-velocity.md)
<br/>  Poorly designed systems become increasingly difficult to modify, slowing down feature development.
- [Brittle Codebase](brittle-codebase.md)
<br/>  Without proper design, changes in one area frequently break other parts of the system.
## Causes ▼

- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Junior developers who have not yet learned software design principles naturally lack design skills.
- [Inadequate Mentoring Structure](inadequate-mentoring-structure.md)
<br/>  Without mentoring, developers do not receive guidance needed to develop design skills over time.
- [Skill Development Gaps](skill-development-gaps.md)
<br/>  Organizations that do not invest in training leave developers without opportunities to build design competencies.
- [Deadline Pressure](deadline-pressure.md)
<br/>  Constant deadline pressure prevents developers from learning and applying proper design practices.
## Detection Methods ○
- **Code Reviews:** Code reviews are a great way to identify design problems.
- **Code Complexity Metrics:** Use static analysis tools to measure the complexity of the codebase.
- **Developer Surveys:** Ask developers about their confidence in their design skills.
- **Architectural Assessments:** Conduct an assessment of the system's architecture to identify design flaws.

## Examples
A company hires a team of junior developers to build a new web application. The developers are not experienced in software design, and they do not have a mentor to guide them. As a result, they create a system that is poorly designed and difficult to maintain. The company is not able to deliver new features in a timely manner, and they are constantly struggling with bugs. The company eventually has to hire a team of experienced developers to rewrite the entire system.
