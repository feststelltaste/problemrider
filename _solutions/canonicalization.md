---
title: Canonicalization
description: Transform input data into a canonical representation
category:
- Security
- Code
quality_tactics_url: https://qualitytactics.de/en/security/canonicalization
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- input-validation
- inconsistent-behavior
- buffer-overflow-vulnerabilities
- log-injection-vulnerabilities
- inadequate-error-handling
layout: solution
---

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

## Examples

> The following scenarios illustrate how canonicalization prevents security bypasses in legacy systems.

A legacy web application has an input filter that blocks SQL injection by checking for the string "SELECT" in form submissions. An attacker bypasses this filter by submitting the query with URL-encoded characters: "%53ELECT". The application's web server decodes the URL encoding before passing it to the application, so the application processes "SELECT" while the filter saw "%53ELECT" and allowed it through. The team implements canonicalization by adding a middleware layer that fully decodes all URL encoding, HTML entities, and Unicode escapes before the input reaches the security filter. After canonicalization, the filter sees "SELECT" regardless of how the attacker encodes it, and the injection attempt is blocked. The team also replaces the simple string-matching filter with parameterized queries, using canonicalization as an additional defense layer.

A legacy file sharing application allows users to download files by specifying a file path parameter. The application checks that the path does not contain ".." to prevent directory traversal. An attacker uses the URL-encoded form "%2e%2e%2f" to traverse directories and access the system's password file. After implementing path canonicalization that resolves all encoded sequences and converts paths to their absolute canonical form before the security check, the application correctly identifies the traversal attempt and rejects it. The canonicalized path "/var/data/../../etc/passwd" becomes "/etc/passwd", which clearly fails the check that all accessed files must be under "/var/data/".
