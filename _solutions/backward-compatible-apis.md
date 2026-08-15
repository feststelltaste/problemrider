---
title: Backward Compatible APIs
description: Evolving API contracts without breaking existing consumers
category:
- Architecture
problems:
- breaking-changes
- legacy-api-versioning-nightmare
- api-versioning-conflicts
- poor-interfaces-between-applications
- integration-difficulties
- fear-of-breaking-changes
layout: solution
related_solutions:
- slug: backward-compatibility
  similarity: 0.9
- slug: forward-compatibility
  similarity: 0.8
- slug: backward-compatible-data-formats
  similarity: 0.8
- slug: api-deprecation-policy
  similarity: 0.8
- slug: api-first-development
  similarity: 0.75
- slug: api-versioning-strategy
  similarity: 0.75
---

## Description

Backward-compatible APIs are interface contracts that evolve only by addition — new optional fields, new endpoints, new response attributes — while existing fields, endpoints, and status codes retain their original meaning and behavior indefinitely, so that clients written against an older version of the contract continue to function unmodified against a newer one. The mechanism relies on both sides holding up their end: the server must never repurpose or remove what already exists, and consumers must act as tolerant readers that ignore fields they do not recognize rather than failing on them. This discipline is especially relevant to legacy systems because their APIs frequently accumulated consumers over many years — internal services, partner integrations, batch jobs — many of which are poorly documented or entirely unknown to the current team, making a coordinated breaking-change rollout across all of them practically impossible. Contract tests that encode old consumers' expectations act as a guardrail, catching accidental breaking changes before they reach production rather than after a partner integration silently fails. The tradeoff is that the API accumulates deprecated fields and dual code paths over time, since nothing is ever cleanly removed without a separate, deliberate deprecation cycle — a cost legacy teams accept because it is smaller than the cost of breaking integrations they cannot even fully enumerate.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Add new fields and endpoints rather than modifying or removing existing ones
- Make new request fields optional with sensible defaults so existing clients do not need to send them
- Use tolerant readers: consumers should ignore unknown fields rather than failing on them
- Apply contract tests that validate old consumer expectations still hold after changes
- Avoid changing the semantic meaning of existing fields or status codes
- When a field must change type or meaning, introduce a new field and deprecate the old one

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables continuous API evolution without coordinated consumer releases
- Reduces integration failures and production incidents caused by breaking changes
- Builds consumer confidence and simplifies partner onboarding

**Costs and Risks:**
- APIs accumulate deprecated fields and endpoints, increasing cognitive load for new developers
- Tolerant reader patterns can hide real bugs in data exchange
- Maintaining backward-compatible behavior in business logic adds implementation complexity
- Eventually requires cleanup through a formal deprecation cycle

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An insurance platform needed to change its claim submission API to support a new document format. Instead of modifying the existing document field, the team added a new optional field for the structured format while keeping the original field functional. Existing consumers continued submitting claims unchanged, while new consumers could opt into the richer format. Consumer-reported errors dropped to zero during the transition, compared to three major incidents during a previous breaking API change.
