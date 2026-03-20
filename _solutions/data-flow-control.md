---
title: Data Flow Control
description: Control and filter data flows between components and systems
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/security/data-flow-control
problems:
- insecure-data-transmission
- data-protection-risk
- authorization-flaws
- cross-system-data-synchronization-problems
- poor-interfaces-between-applications
- cascade-failures
- error-message-information-disclosure
layout: solution
---

## How to Apply ◆

> Legacy systems often pass data freely between components without filtering, validation, or access control, creating opportunities for data leakage, injection attacks, and unauthorized access to sensitive information. Data flow control establishes explicit rules for what data can move between which components and in what form.

- Map all data flows in the legacy system: identify where data originates, what transformations it undergoes, which components it passes through, and where it is stored. Pay special attention to flows that cross trust boundaries (internal to external, application to database, user-facing to backend).
- Classify data by sensitivity level and define handling rules for each classification. Sensitive data (PII, financial records, health information) should be identified and tracked as it flows through the system to ensure appropriate protection at every stage.
- Implement data filtering at component boundaries to strip fields that the receiving component does not need. Legacy APIs often return entire database records when the consumer only needs a few fields, unnecessarily exposing sensitive data.
- Add data validation and sanitization at every trust boundary crossing. Data that enters from an untrusted source must be validated before it is processed, and data that exits to an untrusted destination must be sanitized to prevent information leakage.
- Implement data masking or tokenization for sensitive fields that pass through intermediate components that do not need the actual values. For example, a logging system should receive masked credit card numbers, not full numbers.
- Use network-level controls (firewalls, network policies, service mesh) to enforce permitted data flow paths and prevent unauthorized direct connections between components that should only communicate through defined interfaces.
- Audit data flows periodically to ensure they match the documented flow map and that no unauthorized data paths have been created through configuration changes or workarounds.

## Tradeoffs ⇄

> Data flow control minimizes data exposure and enforces the principle of least privilege at the data level, but it requires comprehensive flow mapping and ongoing governance.

**Benefits:**

- Reduces the blast radius of data breaches by ensuring that each component only has access to the data it needs, limiting what can be exfiltrated from any single point.
- Prevents sensitive data from leaking into logs, error messages, caches, and other locations where it should not appear.
- Supports compliance with data protection regulations (GDPR, HIPAA) that require demonstrable control over how personal data flows and is processed.
- Makes the system's data architecture visible and auditable, enabling informed security decisions.

**Costs and Risks:**

- Mapping all data flows in a complex legacy system is time-consuming and the resulting map requires ongoing maintenance as the system evolves.
- Overly restrictive data flow controls can break existing functionality that depends on access to data that is no longer available after filtering.
- Data masking and tokenization add processing overhead and complexity, particularly when downstream components occasionally need the original values.
- Legacy integration patterns (shared databases, flat-file exchanges) make it difficult to enforce data flow controls at component boundaries.

## How It Could Be

> The following scenarios illustrate how data flow control prevents data exposure in legacy systems.

A legacy e-commerce system logs the full HTTP request body for debugging purposes, including customer credit card numbers and CVV codes in payment requests. A security audit reveals that three years of credit card data are stored in plain text in application log files accessible to 40 developers. The team implements data flow controls by adding a logging middleware that masks sensitive fields (credit card numbers show only the last four digits, CVV codes are replaced with asterisks) before they reach the logging system. They also implement a data flow policy that prohibits PII from flowing into any logging, caching, or analytics system without masking. The existing log files containing unmasked credit card data are securely purged, and automated scanning is added to detect any future instances of sensitive data appearing in logs.

A legacy HR system provides an API consumed by both the company's internal directory and a third-party benefits provider. The API returns the same complete employee record to both consumers, including salary information, Social Security numbers, and performance review data. The benefits provider only needs name, date of birth, and benefits enrollment status. The team implements data flow control by creating consumer-specific API views: the internal directory receives a filtered response containing only name, department, and contact information, while the benefits provider receives only the fields required for benefits administration. Social Security numbers are tokenized in transit and can only be resolved by authorized components. This reduces the sensitive data exposure surface from 45 fields to 3-5 fields per consumer.
