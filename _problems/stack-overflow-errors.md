---
title: Stack Overflow Errors
description: Programs exceed the allocated stack space due to excessive recursion
  or large local variables, causing application crashes.
category:
- Code
- Performance
related_problems:
- slug: memory-leaks
  similarity: 0.55
- slug: resource-allocation-failures
  similarity: 0.55
layout: problem
---

## Description

Stack overflow errors occur when a program's call stack exceeds the allocated stack space, typically due to unbounded recursion, excessively deep function call chains, or allocation of very large local variables. The stack is a limited memory region used for function calls, local variables, and return addresses. When this space is exhausted, the program crashes with a stack overflow error, which can be difficult to debug and may indicate fundamental algorithmic or architectural problems.

## Indicators ⟡

- Application crashes with stack overflow or stack space exceeded errors
- Crashes occur during recursive operations or deeply nested function calls
- Performance degrades before crashes due to excessive stack usage
- Stack traces show very deep call hierarchies or infinite recursion patterns
- Memory profiling shows rapid stack growth during specific operations

## Symptoms ▲

- [System Outages](system-outages.md)
<br/>  Stack overflow errors crash the application, potentially causing outages for users.

## Causes ▼
- [Spaghetti Code](spaghetti-code.md)
<br/>  Tangled code with unpredictable call chains can create deep or circular call hierarchies that exhaust the stack.
- [Complex and Obscure Logic](complex-and-obscure-logic.md)
<br/>  Overly complex recursive logic without proper termination conditions leads to unbounded recursion.

## Detection Methods ○

- **Stack Usage Monitoring:** Monitor stack usage during application execution to identify growth patterns
- **Recursion Depth Tracking:** Instrument recursive functions to track maximum recursion depth
- **Static Analysis:** Analyze code for potential unbounded recursion or large stack allocations
- **Stress Testing:** Test with inputs that may cause deep recursion or large stack usage
- **Stack Trace Analysis:** Examine crash stack traces to identify recursion patterns
- **Profiling Tools:** Use memory profilers to monitor stack usage during operation

## Examples

A file system directory traversal function uses recursion to explore nested folders but lacks a maximum depth limit. When processing a directory structure with hundreds of nested levels (either legitimate or created maliciously), the recursive calls exhaust the stack space and crash the application. Another example involves a mathematical calculation function that recursively computes factorials without checking for reasonable input bounds. When a user inputs a large number like 50000, the recursive factorial calculation creates tens of thousands of stack frames and crashes with a stack overflow before completing the computation.
