---
title: Design by Contract
description: Specifying preconditions, postconditions, and invariants for explicit, verifiable behavior
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/design-by-contract/
problems:
- hidden-side-effects
- assumption-based-development
- implementation-rework
- defensive-coding-practices
- circular-references
- difficult-to-understand-code
- suboptimal-solutions
- ripple-effect-of-changes
- complex-implementation-paths
- cognitive-overload
layout: solution
---

## How to Apply ◆

> In legacy systems, Design by Contract replaces the implicit assumptions that have accumulated over years of undocumented development with explicit, verifiable agreements about what each component expects, guarantees, and maintains. This makes the hidden rules of the legacy codebase visible and enforceable.

- Start by documenting contracts for the most frequently misunderstood functions in the legacy codebase — the ones developers consistently call incorrectly or that produce unexpected side effects. Express preconditions (what must be true before the function is called), postconditions (what the function guarantees after execution), and invariants (what remains true throughout the function's execution).
- Use assertion libraries or language-native contract mechanisms (Java `assert`, Python `assert`, C# Code Contracts, or libraries like `icontract` for Python or `valid4j` for Java) to make contracts executable rather than purely documentary. Executable contracts catch violations during testing and development before they reach production.
- Replace excessive defensive coding with explicit precondition checks at system boundaries. Instead of every internal function validating all its inputs against impossible conditions, validate inputs once at the entry point and use contracts to guarantee that internal functions receive valid data. This eliminates the defensive clutter that obscures business logic.
- Define postconditions that make function behavior explicit, directly addressing the hidden side effects problem. If a function modifies state beyond its return value, the postcondition should declare this explicitly. If a postcondition cannot be cleanly stated because the function does too many things, that is a signal to separate the function into focused components.
- Use class invariants to prevent circular reference problems by declaring the valid state that objects must maintain. For example, an invariant stating "a Document's Page references must point back to this Document and no other" makes the ownership relationship explicit and detectable when violated.
- Apply contracts to interfaces between legacy system modules to define the agreements that were previously implicit. When module A calls module B, the contract specifies exactly what A must provide and what B will deliver, making the ripple effect of changes visible: if a contract changes, all callers must be updated.
- Introduce contracts gradually by targeting code that is actively being modified rather than attempting to retrofit contracts across the entire legacy codebase. Every time a developer touches a function, they add the contract that documents the function's actual behavior.
- Use contract violations discovered during testing as a diagnostic tool: they reveal assumptions that were embedded in the legacy code but never documented, often pointing to the root cause of long-standing bugs.

## Tradeoffs ⇄

> Design by Contract makes the implicit rules of a legacy system explicit and verifiable, but it requires investment in defining and maintaining contracts alongside the code they protect.

**Benefits:**

- Eliminates hidden side effects by forcing developers to declare everything a function does as part of its postcondition, making undocumented behavior visible and subject to review.
- Reduces implementation rework by catching incorrect assumptions early: when a developer's understanding of how a function should be called is wrong, the precondition violation surfaces immediately during development rather than after deployment.
- Replaces wasteful defensive coding with targeted boundary validation, reducing code verbosity and cognitive load while maintaining or improving correctness guarantees.
- Makes the assumptions embedded in legacy code explicit and documented, so developers who join the team later can understand component behavior from contracts rather than by reading and reverse-engineering implementation details.
- Provides precise change impact documentation: when a contract changes, the set of affected callers is immediately identifiable, transforming the unpredictable ripple effect into a bounded, manageable scope.

**Costs and Risks:**

- Contracts add maintenance overhead: when implementation changes, contracts must be updated in sync, and stale contracts are worse than no contracts because they provide false assurance.
- Runtime assertion checking has a performance cost that may be unacceptable in performance-critical legacy code paths; contracts in such areas may need to be disabled in production and only active during testing.
- Legacy code with deeply entangled behavior may have contracts that are extremely complex to specify correctly, and incorrect contracts create a false sense of safety while allowing real bugs to pass.
- Teams unfamiliar with Design by Contract may write trivially obvious contracts (precondition: parameter is not null) that add clutter without value, rather than the meaningful behavioral contracts that provide real protection.
- Retrofitting contracts to legacy code requires understanding the code's actual behavior, which is the very problem contracts are meant to solve — bootstrapping contracts in poorly understood code requires careful characterization testing first.

## How It Could Be

> The following scenarios illustrate how Design by Contract has been applied to bring clarity and correctness guarantees to legacy systems where implicit assumptions caused recurring problems.

A payment processing company had a `TransactionProcessor.process()` method that developers called in several different contexts, each with different assumptions about the state of the transaction object. Some callers expected the method to validate the transaction first; others pre-validated and assumed the method would skip validation. Neither assumption was documented, and the method's behavior depended on an internal flag that was not part of its public interface. The team added explicit preconditions: `transaction.status == VALIDATED` and `transaction.amount > 0`. Postconditions specified: `transaction.status == PROCESSED` and `auditLog.contains(transaction.id)`. Callers that violated preconditions were immediately identified by assertion failures during integration testing, revealing three code paths that had been silently processing invalid transactions for months.

An insurance company's legacy claims system had persistent issues with circular references between `Claim` and `Policy` objects. Both held mutable references to each other, and depending on the order of operations, a claim could end up referencing a policy that referenced a different claim. The team introduced class invariants on both objects: a `Claim` invariant stated that `this.policy.claims.contains(this)` must always hold, and a `Policy` invariant stated that for each claim in its collection, `claim.policy == this`. These invariants immediately flagged the initialization sequence that created orphaned references, a bug that had caused intermittent report inconsistencies for two years.

A logistics company suffered from chronic implementation rework because developers made assumptions about shipment state transitions that did not match actual business rules. A developer would implement "mark shipment as delivered" assuming the shipment was in "in transit" state, but some shipments could be delivered directly from the warehouse without ever transitioning through "in transit." The team defined a state machine contract for the `Shipment` class with explicit preconditions for each state transition. The contract documented that `markDelivered()` required `status in [IN_TRANSIT, AT_WAREHOUSE]`, not just `IN_TRANSIT`. With these contracts in place, developers could see exactly which transitions were valid before writing a single line of implementation code, reducing rework from incorrect state assumptions by over 60%.
