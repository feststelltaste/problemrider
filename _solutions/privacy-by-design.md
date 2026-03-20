---
title: Privacy by Design
description: Embedding privacy protection into system architecture from inception
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/security/privacy-by-design
problems:
- data-protection-risk
- regulatory-compliance-drift
- authentication-bypass-vulnerabilities
- insecure-data-transmission
- poor-documentation
- fear-of-change
- insufficient-audit-logging
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Conduct a data inventory to identify all personal data collected, stored, and processed by the legacy system
- Classify data by sensitivity level and apply appropriate encryption, anonymization, or pseudonymization techniques
- Implement data minimization by removing unnecessary data collection points from legacy forms and APIs
- Add consent management mechanisms and audit trails for data processing activities
- Introduce data retention policies with automated purging of expired personal data
- Retrofit access controls to ensure personal data is only accessible to authorized roles
- Document data flows between legacy components and third-party integrations to identify privacy risks

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces regulatory risk and potential fines from privacy violations
- Builds user trust by demonstrating commitment to data protection
- Simplifies future compliance efforts by establishing privacy as a foundational concern
- Reduces the attack surface by limiting the amount of sensitive data stored

**Costs and Risks:**
- Retrofitting privacy into legacy systems is significantly more expensive than building it in from the start
- Data minimization may require changes to business processes that rely on historical data
- Anonymization and pseudonymization can complicate debugging and support workflows
- Consent management adds complexity to user-facing interfaces and backend logic

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A healthcare platform built in the early 2000s stored patient data in plaintext across multiple database tables with no access controls beyond application-level authentication. When GDPR took effect, the team conducted a full data audit, discovering personal data in 47 tables across three databases. They introduced column-level encryption for sensitive fields, added role-based access controls, and implemented automated data retention with configurable purge schedules. The project took eight months but reduced their compliance risk exposure from critical to low.
