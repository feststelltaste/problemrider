---
title: Dependency Breaking Techniques
description: Create seams in untestable code — extract, wrap, parameterize, subclass — so that a fragment can be isolated and exercised without the rest of the system.
category:
- Code
- Testing
- Architecture
problems:
- difficult-to-test-code
- monolithic-functions-and-classes
- global-state-and-side-effects
- testing-complexity
- poor-encapsulation
- excessive-class-size
- bloated-class
- over-reliance-on-utility-classes
- tight-coupling-issues
- high-coupling-low-cohesion
- legacy-code-without-tests
- hidden-dependencies
- flaky-tests
- circular-references
- god-object-anti-pattern
- refactoring-avoidance
- test-debt
layout: solution
---

## Description

Dependency breaking techniques are a set of small, mechanical code transformations whose purpose is to create a seam — a place where behavior can be substituted without editing the code that uses it. They exist to resolve the circularity that defines legacy work: the code cannot be tested because of its dependencies, and the dependencies cannot be removed safely without tests. Each technique is deliberately minimal and low-risk, designed to be applied by hand without a test net, because at the moment you need them there is no test net. The distinguishing property is conservatism: extracting a method, adding a parameter, or subclassing to override a single call are changes a careful developer can verify by reading. Once one seam exists, a test can be written through it, and from that point ordinary refactoring becomes available.

## How to Apply ◆

> These are the techniques for the specific situation where instantiating one class drags in a database connection, a message broker, a licence server, and the system clock.

- **Extract and override**: pull the problematic call into its own protected method, then create a test-only subclass that overrides it. This is the most broadly applicable technique and requires changing almost nothing in the original code — the extraction is mechanical and the override lives entirely in the test.
- **Parameterize the constructor or method**: where a class constructs its own collaborator internally, add a parameter that accepts one, defaulting to the original construction. Existing callers are unaffected, and tests can now supply a substitute. Keeping the default preserves behavior for all existing call sites, which is what makes this safe to do without tests.
- **Introduce an interface at the boundary** and have the legacy class implement it. Callers depend on the interface; tests supply a different implementation. This is the standard move for isolating databases, file systems, external services, and the clock.
- **Sprout method or class**: when adding new behavior to a tangled method, write it in a new method or class that is fully testable, and have the legacy code call it. The legacy code gains one line; the new logic is tested from the start. This does not improve the old code, and it stops the new code from becoming part of the problem.
- **Wrap method or class**: to add behavior around existing code — logging, validation, a metric — rename the original and create a wrapper with the old name that calls it. Callers are unchanged, and the wrapper is testable independently.
- **Break out a static or global dependency** by introducing an instance-level indirection: replace direct calls to a static holder with calls to a field that defaults to the static holder. Global state is usually the single largest obstacle to testing legacy code, and this converts it from an obstacle into a substitutable one.
- **Encapsulate the clock, randomness, and the environment** behind interfaces early, and treat every direct call to `now()` or a random generator as a defect. These three dependencies account for a disproportionate share of untestable and intermittently failing behavior.
- Apply the techniques **one at a time and verify by reading**. Each transformation should be small enough that its behavior preservation is self-evident. If you cannot convince yourself by reading that a change is behavior-neutral, it is too large a step.
- Once a seam exists, **write a characterization test through it immediately**, before making any further change. The seam has no value until something is exercising it, and the window in which the code is understood is short.

## Tradeoffs ⇄

> These techniques buy testability at the cost of some added indirection, and applied without direction they produce a codebase full of seams that nothing uses.

**Benefits:**

- Code that was untestable becomes testable, which unlocks every subsequent improvement — refactoring, safe bug fixing, and eventual extraction or replacement.
- The transformations are individually low-risk and reviewable by inspection, so they can be applied to code with no test coverage, which is the situation where they are needed.
- New functionality added via sprout techniques is tested from the beginning, so the proportion of tested code rises even when the legacy portion is never addressed.
- Isolating time, randomness, and external services typically removes a substantial share of intermittent test failures and irreproducible defects.
- Seams introduced for testing frequently become the natural boundaries for later extraction, so the work is not wasted if a strangler-style migration follows.

**Costs and Risks:**

- Each technique adds a layer of indirection. Applied liberally without a target, the result is a codebase that is harder to read than the original while being only marginally better tested.
- Extract-and-override in particular produces test-only subclasses and protected methods that exist solely for testing, which some teams find distasteful and which do leak test concerns into production code.
- Without tests, every transformation carries a small residual risk of behavior change, and the risk accumulates across many transformations. Discipline about step size is essential.
- The techniques address structure, not design. A class made testable is not thereby well-designed, and teams sometimes stop at testability and consider the module addressed.
- Legacy languages and frameworks vary in how well they support these moves; some make substitution genuinely difficult, and the effort can exceed the value for a module that rarely changes.

## How It Could Be

A developer needed to fix a rounding defect in an order total calculation. The calculating class constructed a database connection, read a static currency configuration, and called the system clock for the exchange rate date — instantiating it in a test was impossible. Rather than attempting to restructure the class, she applied three transformations in one afternoon: the exchange rate lookup was extracted into a protected method and overridden in a test subclass, the currency configuration was replaced with a field defaulting to the static holder, and the clock call became a constructor parameter with a default. None of the changes altered production behavior, and all three were verifiable by reading. She then wrote eleven characterization tests, found that the rounding defect was one of three, and fixed all three with confidence.

A team adding a new fraud check to a payment flow chose to sprout rather than modify. The existing payment method was 700 lines with no tests; instead of extending it, they wrote a `FraudCheck` class with full test coverage and inserted a single call into the legacy method. The legacy method grew by one line and remained as untested as before, but the new logic — the part most likely to need changing as fraud patterns evolved — was properly tested from day one. Over the following two years the fraud check was modified fourteen times, always safely, while the surrounding legacy method was never touched.
