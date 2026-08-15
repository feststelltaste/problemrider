---
title: Living Documentation
description: Current and easily accessible documentation as an integral part of development
category:
- Communication
- Process
problems:
- poor-documentation
- information-decay
- legacy-system-documentation-archaeology
- implicit-knowledge
- knowledge-silos
- difficult-developer-onboarding
- tacit-knowledge
- unclear-documentation-ownership
- accumulated-decision-debt
- duplicated-research-effort
- extended-research-time
- information-fragmentation
- knowledge-sharing-breakdown
- team-churn-impact
- incomplete-knowledge
layout: solution
related_solutions:
- slug: documentation-as-code
  similarity: 0.85
- slug: architecture-documentation
  similarity: 0.8
- slug: documentation-of-compatibility-requirements
  similarity: 0.75
- slug: runbooks
  similarity: 0.75
- slug: pattern-language
  similarity: 0.75
- slug: architecture-decision-records
  similarity: 0.75
---

## Description

Living documentation is documentation that is generated from, embedded in, or continuously verified against the system it describes, so that it evolves automatically alongside the code rather than drifting away from it the moment it is written. In practice this means storing documentation as version-controlled text next to the code it documents, generating API references directly from code annotations or OpenAPI specifications, writing architecture decision records at the moment decisions are made rather than after the fact, and using executable specifications that fail a build the instant the described behavior and the actual behavior diverge. Legacy systems are the environment where conventional documentation fails hardest: wikis and Word documents are written once during an initial project phase and then abandoned, while the system itself keeps changing for years, so by the time a new maintainer needs the documentation it describes a system that no longer exists, and the resulting distrust means people stop consulting it and stop updating it, accelerating the decay. Living documentation breaks this cycle by removing the manual synchronization step entirely — because the documentation is either derived mechanically from the code or actively enforced by tests, it cannot silently fall out of date without the build itself signaling the discrepancy. This matters most in legacy modernization work, where institutional knowledge about why the system behaves as it does is often concentrated in a few people who may leave, and where new team members otherwise have no reliable way to distinguish current, accurate context from years-old, superseded notes.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Store documentation alongside the code in version control so it evolves with the system
- Use executable specifications (e.g., BDD-style tests) that serve as both documentation and verification
- Generate API documentation from code annotations or OpenAPI specifications to keep it always current
- Adopt architecture decision records (ADRs) to capture the reasoning behind key design choices
- Integrate documentation checks into the CI pipeline to detect stale or broken references
- Replace static wiki pages with documentation-as-code approaches that are reviewed in pull requests
- Start with the areas of the legacy system that cause the most confusion or onboarding friction

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Documentation stays accurate because it is updated as part of the normal development workflow
- Reduces onboarding time by providing discoverable, up-to-date system knowledge
- Executable specifications provide both documentation and regression protection
- Version-controlled documentation preserves history and enables blame tracking

**Costs and Risks:**
- Requires cultural change: teams must treat documentation as a first-class deliverable
- Initial effort to establish the infrastructure and migrate existing documentation
- Executable specifications add maintenance overhead when system behavior changes frequently
- Risk of documentation bloat if there is no curation or pruning discipline

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An insurance company inherited a legacy claims processing system where the only documentation was a set of Word documents last updated five years prior. New developers spent weeks asking colleagues to understand business rules. The team began writing ADRs for every significant change, added Javadoc-generated API references, and introduced Cucumber scenarios that described the claims workflow in business language. Within six months, onboarding time dropped from three weeks to one, and the Cucumber scenarios caught several cases where the documentation contradicted actual system behavior, leading to important bug discoveries.
