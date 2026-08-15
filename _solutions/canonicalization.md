---
title: Canonicalization
description: Transform input data into a canonical representation
category:
- Security
- Code
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- inconsistent-behavior
- buffer-overflow-vulnerabilities
- log-injection-vulnerabilities
- inadequate-error-handling
layout: solution
related_solutions:
- slug: input-validation
  similarity: 0.8
- slug: output-encoding
  similarity: 0.75
- slug: authentication
  similarity: 0.7
- slug: encryption
  similarity: 0.7
- slug: data-flow-control
  similarity: 0.7
- slug: defense-lines
  similarity: 0.65
---

## Description

Canonicalization is the process of transforming input into a single, well-defined standard representation before that input is validated, compared, or acted upon, so that security checks and business logic operate on one predictable form instead of any of its many equivalent encodings. Data frequently arrives with the same underlying meaning expressed in different ways — URL-encoded or double-encoded characters, distinct Unicode normalization forms, path expressions with redundant separators or traversal sequences — and canonicalization collapses these variants into one form before anything downstream inspects them. This matters acutely for legacy systems because their input filters and validation routines were often written for a single encoding assumption and never updated as new encoding paths were added at the network or application layer, leaving a gap between what a filter inspects and what the application actually processes. Attackers exploit precisely that gap, submitting a payload in an encoding the filter does not recognize as dangerous, counting on a downstream component to decode it into the harmful form after the check has already passed. By normalizing first and validating the canonical form rather than the raw input, canonicalization closes this bypass class systematically rather than requiring every filter to anticipate every possible encoding trick. It also simplifies validation logic itself, since rules only need to account for one representation instead of an open-ended set of equivalent ones, which is valuable in legacy codebases where validation is often duplicated inconsistently across many entry points. Because canonicalization can alter data if applied incorrectly, it has to be implemented with care in systems whose internal logic already relies on specific non-canonical forms.

## How to Apply ◆

> Legacy systems often process input in multiple encodings and formats without normalizing them first, creating opportunities for attackers to bypass security filters using encoded or obfuscated payloads. Canonicalization transforms all input into a single standard form before validation and processing.

- Identify all input entry points in the legacy system where data arrives in variable formats: URLs, file paths, character encodings, Unicode representations, HTML entities, and URL-encoded values.
- Apply canonicalization as the first step in input processing, before any security checks or validation. Validate against the canonical form, not the raw input — attackers exploit the gap between what the security filter sees and what the application processes.
- Normalize Unicode input to a consistent form (NFC or NFKC) to prevent attacks using visually identical but technically different character sequences. Legacy systems often do not handle Unicode normalization, allowing homograph attacks and filter bypasses.
- Resolve all path components (dot-dot sequences, symbolic links, redundant separators) to absolute canonical paths before checking access permissions. This prevents path traversal attacks that use encoded directory traversal sequences.
- Decode all encoding layers (URL encoding, HTML entities, Base64, double encoding) completely before applying validation rules. Many legacy security filters check only the first encoding layer while the application decodes multiple layers.
- Standardize data formats (dates, numbers, identifiers) into a single canonical representation at the system boundary to prevent inconsistencies that lead to logic errors and security bypasses.
- Implement canonicalization in a shared utility library so all input processing paths use the same normalization logic, preventing inconsistencies between different parts of the codebase.

## Tradeoffs ⇄

> Canonicalization eliminates encoding-based security bypasses by ensuring all input is in a known, standard form before validation, but it requires comprehensive identification of all encoding schemes and careful implementation.

**Benefits:**

- Prevents security filter bypasses using alternative encodings, double encoding, and Unicode tricks that exploit differences between the filter's view and the application's view of input.
- Reduces the complexity of validation rules by ensuring they only need to handle one canonical form rather than multiple equivalent representations.
- Improves data consistency by normalizing inputs to a standard form at the system boundary.
- Makes security testing more effective because the canonical form is predictable and can be systematically validated.

**Costs and Risks:**

- Incorrect canonicalization can alter the semantic meaning of input, causing data corruption or functional errors.
- Legacy systems may rely on specific non-canonical representations internally, making canonicalization at the boundary incompatible with existing processing logic.
- Over-aggressive canonicalization (stripping or replacing characters) can reject or corrupt legitimate international input.
- Performance overhead from canonicalization is typically small but can be noticeable for high-volume input processing in legacy systems.

## How It Could Be

> The following scenarios illustrate how canonicalization prevents security bypasses in legacy systems.

A legacy web application has an input filter that blocks SQL injection by checking for the string "SELECT" in form submissions. An attacker bypasses this filter by submitting the query with URL-encoded characters: "%53ELECT". The application's web server decodes the URL encoding before passing it to the application, so the application processes "SELECT" while the filter saw "%53ELECT" and allowed it through. The team implements canonicalization by adding a middleware layer that fully decodes all URL encoding, HTML entities, and Unicode escapes before the input reaches the security filter. After canonicalization, the filter sees "SELECT" regardless of how the attacker encodes it, and the injection attempt is blocked. The team also replaces the simple string-matching filter with parameterized queries, using canonicalization as an additional defense layer.

A legacy file sharing application allows users to download files by specifying a file path parameter. The application checks that the path does not contain ".." to prevent directory traversal. An attacker uses the URL-encoded form "%2e%2e%2f" to traverse directories and access the system's password file. After implementing path canonicalization that resolves all encoded sequences and converts paths to their absolute canonical form before the security check, the application correctly identifies the traversal attempt and rejects it. The canonicalized path "/var/data/../../etc/passwd" becomes "/etc/passwd", which clearly fails the check that all accessed files must be under "/var/data/".
