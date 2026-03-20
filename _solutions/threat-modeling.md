---
title: Threat Modeling
description: Conduct systematic analysis of threats, attackers, and countermeasures
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/security/threat-modeling
problems:
- implementation-starts-without-design
- quality-blind-spots
- architectural-mismatch
- authentication-bypass-vulnerabilities
- authorization-flaws
- system-integration-blindness
- stagnant-architecture
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Create data flow diagrams of the legacy system identifying all entry points, data stores, and trust boundaries
- Apply a structured methodology such as STRIDE or PASTA to systematically identify threats at each component
- Identify potential attackers, their motivations, and capabilities relevant to the system
- Rank identified threats by risk level considering both likelihood and business impact
- Define countermeasures for each threat and map them to existing or planned security controls
- Update threat models when the system architecture changes or new threat information becomes available
- Involve both security specialists and developers with deep legacy system knowledge in the modeling process

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides structured identification of security risks that ad-hoc approaches miss
- Focuses security investment on the most impactful threats rather than spreading effort uniformly
- Creates shared understanding of security risks between development and security teams
- Produces documentation that supports security decision making and compliance requirements

**Costs and Risks:**
- Threat modeling requires significant time investment from experienced practitioners
- Legacy systems with poor documentation make accurate threat modeling difficult
- Models can become outdated quickly if not maintained alongside system changes
- Incomplete threat models can create false confidence about security coverage
- Analysis paralysis can occur if threat modeling becomes too detailed or academic

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A bank conducted its first threat model for a legacy wire transfer system that had been in production for 18 years. The STRIDE analysis of the system's data flow diagrams revealed that an internal API used for batch processing accepted unauthenticated requests from any host on the internal network, an assumption that was reasonable in 2006 but dangerous given the current threat landscape. The threat model also identified that the system's logging was insufficient to detect or investigate transaction manipulation. These findings drove targeted security improvements that addressed the highest-risk threats without requiring a full system rewrite.
