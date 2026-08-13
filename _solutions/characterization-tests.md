---
title: Characterization Tests
description: Capture what legacy code currently does — correct or not — as executable tests, creating a safety net for changing code whose intended behavior is unknown.
category:
- Testing
- Code
problems:
- legacy-code-without-tests
- poor-test-coverage
- outdated-tests
- testing-complexity
- fear-of-breaking-changes
- fear-of-change
- delayed-bug-fixes
- partial-bug-fixes
- increased-manual-testing-effort
- defensive-coding-practices
- legacy-business-logic-extraction-difficulty
- difficult-to-test-code
- maintenance-paralysis
- regression-bugs
- flaky-tests
- strangler-fig-pattern-failures
- global-state-and-side-effects
- increased-bug-count
- refactoring-avoidance
- test-debt
- cache-invalidation-problems
- hidden-side-effects
- history-of-failed-changes
- increasing-brittleness
- monolithic-functions-and-classes
- brittle-codebase
- entity-attribute-value-overuse
layout: solution
---

## Description

A characterization test records what a piece of code actually does, rather than what it is supposed to do. You call the code with a set of inputs, observe the output, and assert that the output stays the same. This deliberately encodes existing bugs as expected behavior, which is unsettling at first and is exactly the point: for legacy code with no specification, no documentation, and no surviving author, the current behavior is the only specification that exists, and downstream systems have been depending on it — including its defects — for years. Characterization tests are not a substitute for proper tests; they are the scaffolding that makes it safe to work on code long enough to eventually write proper ones. Their function is to convert "I don't know what this does" from a reason not to touch the code into a documented, executable fact.

## How to Apply ◆

> The typical target is a module that everyone avoids because nobody knows what depends on its quirks — which is precisely the module that most needs to be changed.

- **Find the smallest seam** through which the code can be invoked. Often it is not a unit-level entry point at all but an HTTP endpoint, a batch job, or a database procedure. Test at whatever level is reachable today; a coarse test that exists beats a fine-grained test that requires six weeks of refactoring first.
- **Write a test that asserts something you know to be wrong**, run it, and let it fail. The failure message tells you the actual value. Paste that value into the assertion. This sounds crude and is the fastest reliable way to characterize behavior you cannot predict by reading.
- Use **approval testing** for outputs too large or complex to assert field by field: serialize the entire output to a file, review it once by hand, and commit it as the approved baseline. Subsequent runs diff against it. This is the practical approach for report generators, document producers, and message transformers.
- **Generate input coverage systematically** rather than by intuition. Boundary values, nulls, empty collections, and — where available — a sample of real production inputs will exercise paths that hand-written examples miss. Recording a set of real production requests and replaying them is often the highest-value single step.
- Use **coverage measurement as a guide, not a goal**, during the characterization phase. Its purpose here is to reveal which branches your inputs have not reached yet, so you can construct inputs that do. A branch never exercised by any characterization test is a branch you are about to change blind.
- **Mark the tests as characterization tests explicitly** — a naming convention, an annotation, a separate directory. Later readers must be able to tell that these assertions describe observed behavior, not intended behavior, or someone will eventually "fix" a test that is documenting a bug that a downstream system relies on.
- When a characterization test **later reveals an actual bug**, decide deliberately and record the decision: fix it and update the test with a note explaining the change, or keep the behavior and document why it must be preserved. Do not silently change either.
- **Convert to specification tests progressively.** As the intended behavior of a region becomes understood, replace characterization assertions with tests that state the requirement, and delete the redundant baselines. The characterization suite should shrink over the life of a modernization effort.
- Accept that the suite will be **ugly and repetitive**. Characterization tests are tooling for a transitional period, not a codebase to be proud of, and time spent making them elegant is usually better spent extending their coverage.

## Tradeoffs ⇄

> Characterization tests provide a safety net quickly and cheaply, at the price of a test suite that documents defects as requirements and must be actively unwound later.

**Benefits:**

- Refactoring becomes possible in code that could not previously be touched safely, which is the precondition for essentially every other improvement to a legacy module.
- Existing behavior — including undocumented behavior that consumers depend on — is preserved through changes, which is the main risk in legacy modification and the hardest to reason about without tests.
- Undocumented business rules become visible and readable, since the tests are an executable description of what the system actually does. They frequently become the first real documentation of a module.
- Bugs are discovered as a side effect: writing inputs that exercise every branch regularly surfaces behavior that nobody intended and nobody had noticed.
- The tests can be written by someone who does not understand the domain, which matters when the original developers are gone and no domain expert is available.

**Costs and Risks:**

- The suite encodes current bugs as expected behavior. Without clear marking, later developers will treat these assertions as requirements and preserve defects indefinitely.
- Characterization tests are brittle by construction: they fail on any behavior change, including intended ones, which produces noise and can lead to the suite being ignored or bulk-updated without review.
- Coarse-grained tests through the outermost seam are slow, and a slow suite gets run less often, eroding the safety net exactly when changes are being made most rapidly.
- Approval baselines can be updated thoughtlessly. A workflow where regenerating the baseline is easier than understanding the diff defeats the entire mechanism, so baseline changes must be reviewed like code.
- The tests provide no guidance about what the code should do, so they protect against regression while offering no help in deciding what to build.

## How It Could Be

A team needed to change tax calculation logic in a payroll system: 4,000 lines of nested conditionals, no tests, and the original developer retired six years earlier. Rather than reading it, they wrote a harness that ran the calculation against 12,000 anonymized historical payroll records and stored the results as an approved baseline. Constructing the harness took four days. It immediately revealed that eleven records produced different results on repeated runs — a dependency on the system clock that nobody had known about, and which had been quietly producing occasional incorrect deductions for years. With the baseline in place, the team refactored the module over five weeks, running the full comparison after each change. Two of the intermediate changes produced unexpected diffs and were reverted within minutes.

A different team used characterization tests to make a decision rather than to enable a refactoring. They were assessing whether an invoice generator could be replaced by a vendor product. They characterized the existing generator's output across 800 sample invoices, then ran the same inputs through the candidate product and diffed. The comparison found 43 systematic differences, of which 6 turned out to be legal requirements specific to two jurisdictions that the vendor product did not support. That finding, produced in two weeks, prevented a procurement decision that would have been discovered as unworkable roughly nine months into an integration project.
