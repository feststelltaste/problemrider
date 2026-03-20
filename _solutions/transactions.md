---
title: Transactions
description: Grouping multiple operations into an atomic, consistent unit
category:
- Architecture
- Database
quality_tactics_url: https://qualitytactics.de/en/reliability/transactions
problems:
- silent-data-corruption
- race-conditions
- data-migration-integrity-issues
- inconsistent-behavior
- long-running-database-transactions
- long-running-transactions
- deadlock-conditions
- cascade-failures
- synchronization-problems
layout: solution
---

## How to Apply ◆

> Legacy systems often perform multi-step data modifications without transactional guarantees, leaving data in inconsistent states when failures occur midway through an operation. Proper transaction management ensures that related operations either all succeed or all fail as a unit.

- Identify all multi-step operations in the legacy system where partial completion would leave data in an inconsistent state. Common examples include order processing (reserve inventory, charge payment, create shipment), financial transfers (debit one account, credit another), and master data updates that span multiple tables.
- Wrap related database operations in explicit transactions with appropriate isolation levels. Many legacy systems rely on auto-commit mode, where each SQL statement is its own transaction, providing no atomicity across related operations.
- Choose the minimum isolation level that provides correctness for each use case. READ COMMITTED is sufficient for most operations; SERIALIZABLE prevents all anomalies but dramatically reduces concurrency. Legacy systems with high contention suffer disproportionately from over-restrictive isolation levels.
- Implement the saga pattern for operations that span multiple services or databases where a single ACID transaction is not possible. Define compensating transactions for each step so the overall operation can be rolled back if any step fails.
- Keep transaction scope as small as possible — acquire locks late, release them early. Legacy systems frequently hold transactions open during user interactions or external API calls, causing lock contention and deadlocks.
- Add idempotency keys to transaction-triggering operations so that retries after timeout or network failure do not result in duplicate processing. This is critical for legacy systems where the caller cannot reliably determine whether a timed-out transaction committed or rolled back.
- Implement proper error handling that rolls back transactions on any exception, including unexpected runtime errors. Legacy code often catches and swallows exceptions without rolling back, leaving partially committed data.

## Tradeoffs ⇄

> Transactions provide data consistency guarantees that prevent corruption from partial operations, but they introduce contention overhead and complexity in distributed systems.

**Benefits:**

- Prevent data corruption from partial operation completion by ensuring all-or-nothing semantics for related changes.
- Simplify error recovery by providing automatic rollback when any step in a multi-step operation fails.
- Enable concurrent access to shared data with well-defined consistency guarantees through isolation levels.
- Provide a foundation for audit trails and compliance by ensuring that recorded state transitions are always complete and consistent.

**Costs and Risks:**

- Long-running transactions hold locks that block other operations, reducing system throughput and creating contention bottlenecks in legacy systems with shared databases.
- Distributed transactions across multiple databases or services are complex, fragile, and significantly reduce availability, often requiring saga-based alternatives that are harder to implement correctly.
- Deadlocks become more likely as transaction scope increases, requiring detection, timeout, and retry logic.
- Legacy databases may have limited transaction support or unexpected behavior under certain isolation levels, requiring careful testing.
- Transaction retries after failures can cause duplicate side effects (sending emails, calling external APIs) unless idempotency is explicitly designed in.

## How It Could Be

> The following scenarios illustrate how proper transaction management prevents data corruption in legacy systems.

A legacy e-commerce system processes orders by executing a sequence of five SQL statements: insert the order header, insert order line items, decrement inventory, insert a payment record, and update the customer's order count. These statements execute individually with auto-commit enabled. When the database connection drops after the payment record is inserted but before inventory is decremented, the system creates an order with a payment but without reserving inventory. The same items are then sold to another customer, resulting in overselling. The team wraps all five statements in a single database transaction, ensuring that either all succeed or none do. They also add a unique order ID as an idempotency key so that if the client retries after a connection failure, the retry detects the existing transaction and returns the result rather than creating a duplicate order.

A legacy banking system transfers funds between accounts using two separate UPDATE statements — one to debit the source account and one to credit the destination account. Under heavy load, the application occasionally crashes between the debit and credit operations, resulting in money disappearing from the source account without appearing in the destination. The team implements explicit transaction wrapping with SERIALIZABLE isolation for transfer operations. They also discover that the legacy code catches database exceptions and logs them but does not roll back the transaction, leaving partial changes committed. After fixing the error handling to always roll back on failure and implementing automatic retry with exponential backoff for serialization conflicts, the system has zero fund discrepancy incidents over the following year.
