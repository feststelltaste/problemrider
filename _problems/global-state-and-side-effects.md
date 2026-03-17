---
title: Global State and Side Effects
description: Excessive use of global variables or functions with hidden side effects
  makes it difficult to reason about code behavior.
category:
- Architecture
- Code
related_problems:
- slug: hidden-side-effects
  similarity: 0.65
- slug: unpredictable-system-behavior
  similarity: 0.55
- slug: complex-and-obscure-logic
  similarity: 0.55
- slug: spaghetti-code
  similarity: 0.55
- slug: hidden-dependencies
  similarity: 0.55
- slug: inconsistent-codebase
  similarity: 0.5
layout: problem
---

## Description
Global state and side effects are a common source of complexity and bugs in software systems. Global state refers to data that is accessible and mutable from anywhere in the codebase, while side effects are modifications to state that occur as a byproduct of a function call. When used excessively, these constructs can make it very difficult to reason about the behavior of the system, as the impact of a change can be far-reaching and unpredictable.

## Indicators ⟡
- It is difficult to understand the impact of a change to a piece of code.
- The same bug appears in different parts of the system.
- The system behaves differently in different environments, even though the code is the same.

## Symptoms ▲

- [Unpredictable System Behavior](unpredictable-system-behavior.md)
<br/>  Global state mutations from any part of the codebase cause unexpected side effects in seemingly unrelated areas.
- [Hidden Dependencies](hidden-dependencies.md)
<br/>  Components that share global state become implicitly dependent on each other without this being visible in the code structure.
- [Difficult to Test Code](difficult-to-test-code.md)
<br/>  Functions relying on global state cannot be tested in isolation because their behavior depends on externally mutable state.
- [Regression Bugs](regression-bugs.md)
<br/>  Changes to global state in one area break existing functionality elsewhere because the dependencies are not apparent.
- [Race Conditions](race-conditions.md)
<br/>  Mutable global state accessed by multiple threads without synchronization leads to data races and corruption.
- [Debugging Difficulties](debugging-difficulties.md)
<br/>  Tracing bugs is extremely difficult when any part of the codebase can modify shared global state unpredictably.
## Causes ▼

- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers lacking experience with proper software design patterns default to using global variables as the simplest approach.
- [Procedural Programming in OOP Languages](procedural-programming-in-oop-languages.md)
<br/>  A procedural mindset leads developers to use global variables and functions with side effects rather than encapsulated objects.
- [Poor Encapsulation](poor-encapsulation.md)
<br/>  Failure to properly encapsulate state within objects exposes it globally, inviting widespread mutations and side effects.
- [Spaghetti Code](spaghetti-code.md)
<br/>  Tangled, unstructured code naturally gravitates toward global state as a way to share data between poorly organized components.
## Detection Methods ○
- **Static Analysis:** Use static analysis tools to identify the use of global variables and functions with side effects.
- **Code Reviews:** Pay close attention to the use of global state and side effects during code reviews.
- **Testing:** Write tests that expose the hidden dependencies and side effects in the code.

## Examples
A function that calculates the total price of a shopping cart also has a side effect of applying a discount to the user's account. This side effect is not documented, and it is not obvious from the function's name or signature. As a result, a developer who calls this function to simply display the total price in the UI inadvertently applies a discount to the user's account, leading to a loss of revenue for the company. This is a classic example of how hidden side effects can lead to unexpected and undesirable behavior.
