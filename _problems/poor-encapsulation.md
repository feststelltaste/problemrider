---
title: Poor Encapsulation
description: Data and the behavior that acts on that data are not bundled together
  in a single, cohesive unit, leading to a lack of data hiding and a high degree of
  coupling.
category:
- Architecture
- Code
related_problems:
- slug: misunderstanding-of-oop
  similarity: 0.6
- slug: high-coupling-low-cohesion
  similarity: 0.6
- slug: tight-coupling-issues
  similarity: 0.6
- slug: over-reliance-on-utility-classes
  similarity: 0.55
- slug: incomplete-knowledge
  similarity: 0.55
- slug: complex-and-obscure-logic
  similarity: 0.55
solutions:
- incremental-refactoring
layout: problem
---

## Description
Poor encapsulation is a common design problem in object-oriented programming. It occurs when data and the behavior that acts on that data are not bundled together in a single, cohesive unit. This can lead to a number of problems, including a lack of data hiding, a high degree of coupling, and a system that is difficult to understand and maintain. Poor encapsulation is often a sign of a lack of understanding of the principles of object-oriented design.

## Indicators ⟡
- Classes have a large number of public fields.
- Classes have a large number of getters and setters.
- Data is passed around between a large number of different objects.
- It is difficult to understand how data is being used in the system.

## Symptoms ▲

- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  Exposing internal data directly creates dependencies between components, making them tightly coupled.
- [Ripple Effect of Changes](ripple-effect-of-changes.md)
<br/>  When internal state is publicly accessible, changes to data structures propagate throughout the codebase.
- [Difficult to Test Code](difficult-to-test-code.md)
<br/>  Without encapsulation, components are hard to test in isolation because they depend on and expose shared mutable state.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Poor encapsulation makes the system more expensive to maintain because any internal change can break external consumers.
- [Brittle Codebase](brittle-codebase.md)
<br/>  Lack of data hiding makes the codebase fragile, as many components depend on implementation details that may change.
## Causes ▼

- [Misunderstanding of OOP](misunderstanding-of-oop.md)
<br/>  Developers who don't understand object-oriented principles fail to properly encapsulate data and behavior.
- [Procedural Programming in OOP Languages](procedural-programming-in-oop-languages.md)
<br/>  Applying procedural thinking to OOP languages results in data structures without associated behavior.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Less experienced developers often default to public fields and getters/setters without understanding encapsulation benefits.
- [Deadline Pressure](deadline-pressure.md)
<br/>  Under time pressure, developers take shortcuts by making fields public rather than designing proper interfaces.
## Detection Methods ○
- **Code Reviews:** Code reviews are a great way to identify problems with encapsulation.
- **Static Analysis:** Use static analysis tools to identify classes with a large number of public fields or getters and setters.
- **Dependency Analysis:** Analyze the dependencies between the components of the system to identify areas of high coupling.
- **Code Coverage:** Measure the code coverage of your tests. A low code coverage may be a sign of poor encapsulation.

## Examples
A class has a public field that is accessed by a large number of other classes. This is an example of poor encapsulation. The problem could be solved by making the field private and providing a public method for accessing it. This would hide the implementation details of the class and would make it easier to change the class in the future without breaking other parts of the system.
