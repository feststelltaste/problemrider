---
title: Integer Overflow and Underflow
description: Arithmetic operations produce results that exceed the maximum or minimum
  values representable by integer data types, leading to unexpected behavior.
category:
- Code
- Data
- Security
related_problems:
- slug: buffer-overflow-vulnerabilities
  similarity: 0.65
- slug: stack-overflow-errors
  similarity: 0.55
layout: problem
---

## Description

Integer overflow and underflow occur when arithmetic operations produce results that cannot be represented within the bounds of the integer data type being used. Overflow happens when a result exceeds the maximum representable value, while underflow occurs when a result is less than the minimum representable value. In most programming languages, these conditions cause the value to wrap around, leading to unexpected and potentially dangerous behavior including security vulnerabilities.

## Indicators ⟡

- Calculations produce unexpectedly small or negative results from large positive inputs
- Loop counters or array indices become negative unexpectedly
- Security checks are bypassed due to unexpected value wrapping
- Memory allocation fails or allocates incorrect amounts due to size calculation errors
- Financial or measurement calculations produce obviously incorrect results

## Symptoms ▲

- [Buffer Overflow Vulnerabilities](buffer-overflow-vulnerabilities.md)
<br/>  Integer overflow in size calculations can lead to undersized buffer allocations that are then overflowed.
- [Silent Data Corruption](silent-data-corruption.md)
<br/>  Value wrapping from overflow produces incorrect data that may propagate undetected through the system.
- [Unpredictable System Behavior](unpredictable-system-behavior.md)
<br/>  Wrapped values cause unexpected program behavior that is difficult to reproduce and diagnose.
- [Increased Risk of Bugs](increased-risk-of-bugs.md)
<br/>  Integer overflow conditions create subtle bugs that only manifest with specific input values.
## Causes ▼

- [Insufficient Design Skills](insufficient-design-skills.md)
<br/>  Developers without awareness of data type limits fail to implement proper bounds checking.
- [Quality Blind Spots](insufficient-testing.md)
<br/>  Insufficient testing with boundary values fails to detect overflow conditions before production.
- [Inadequate Error Handling](inadequate-error-handling.md)
<br/>  Lack of validation and error handling for arithmetic operations allows overflows to occur silently.
## Detection Methods ○

- **Static Analysis Tools:** Use tools that can identify potential integer overflow conditions in arithmetic operations
- **Runtime Overflow Detection:** Use compiler flags or runtime libraries that detect integer overflow during execution
- **Bounds Checking:** Implement explicit bounds checking before arithmetic operations
- **Input Validation Testing:** Test with extreme input values to identify overflow conditions
- **Code Review:** Review arithmetic operations for potential overflow scenarios
- **Fuzzing:** Use automated testing with large or unusual input values to trigger overflow conditions

## Examples

A web application calculates memory buffer sizes by multiplying the number of items by the size per item. An attacker provides an extremely large item count that causes integer overflow, making the calculated buffer size wrap around to a small positive value. The application allocates a small buffer but then writes data for the original large number of items, causing a buffer overflow that can be exploited for code injection. Another example involves a financial application that calculates interest by multiplying principal amounts by interest rates. For very large principal amounts, the multiplication overflows and wraps to a negative value, causing the application to incorrectly calculate negative interest and potentially crediting money to accounts inappropriately.
