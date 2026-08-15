---
title: Threat Modeling
description: Conduct systematic analysis of threats, attackers, and countermeasures
category:
- Security
- Architecture
problems:
- implementation-starts-without-design
- quality-blind-spots
- architectural-mismatch
- authentication-bypass-vulnerabilities
- authorization-flaws
- system-integration-blindness
- stagnant-architecture
layout: solution
related_solutions:
- slug: security-architecture-analysis
  similarity: 0.85
- slug: risk-analysis
  similarity: 0.8
- slug: security-frameworks
  similarity: 0.8
- slug: security-certification
  similarity: 0.8
- slug: secure-software-development
  similarity: 0.8
- slug: threat-intelligence
  similarity: 0.8
---

## Description

Threat modeling is a structured analytical exercise that maps a system's components, data flows, and trust boundaries, then systematically asks what could go wrong at each point — who might attack it, how, and with what consequence. Methodologies such as STRIDE or PASTA give this process a repeatable checklist instead of relying on whichever risks happen to occur to whoever is in the room, which matters because unaided intuition tends to focus on familiar, recently discussed threats and miss the rest. In legacy systems, threat modeling is especially valuable precisely because the original design rationale is usually gone: assumptions about network trust, user behavior, or deployment topology that were reasonable when the system was built have often quietly become false as the surrounding environment evolved, and nobody currently on the team decided to accept the resulting risk — it simply accumulated unnoticed. Producing an explicit diagram and threat list forces those historical assumptions into the open, where they can be evaluated against the current threat landscape rather than inherited by default. The output also gives security investment a rational basis in an environment where remediation resources are limited and legacy architecture cannot always be redesigned outright, letting teams direct effort at the highest-risk exposures instead of spreading it evenly across a system nobody fully understands anymore.

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
