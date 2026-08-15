---
title: A/B Testing
description: Comparing different versions to optimize user experience
category:
- Testing
- Requirements
problems:
- poor-user-experience-ux-design
- customer-dissatisfaction
- negative-user-feedback
- declining-business-metrics
- user-frustration
- difficulty-quantifying-benefits
- feature-bloat
layout: solution
related_solutions:
- slug: user-centered-design
  similarity: 0.8
- slug: consistent-user-interface
  similarity: 0.75
- slug: adaptive-behavior
  similarity: 0.75
- slug: risk-analysis
  similarity: 0.75
- slug: cross-version-testing
  similarity: 0.75
- slug: usability-tests
  similarity: 0.75
---

## Description

A/B testing is a controlled experiment in which two or more variants of a feature, workflow, or interface are shown to different segments of live users simultaneously, with the resulting behavioral or business metrics compared to determine which variant performs better. Rather than deciding a change is an improvement by internal conviction or aesthetic preference, the method treats every proposed change to the system as a hypothesis to be validated against real usage data before it is rolled out fully. In legacy systems, where years of accumulated assumptions about user behavior are baked into the UI and workflow but rarely revisited, A/B testing offers a way to challenge those assumptions incrementally and safely, without committing to a risky big-bang redesign. It also converts vague complaints about user experience into measurable outcomes, making it possible to prioritize modernization effort by quantified impact rather than by whoever complains loudest. Because legacy codebases are often not built with experimentation in mind, applying this technique typically requires first adding feature flagging and analytics instrumentation as a prerequisite, which is itself a useful forcing function for improving the system's observability and modularity.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify specific user experience hypotheses to test rather than making changes based on assumptions alone
- Implement feature flagging infrastructure that allows controlled rollout of changes to subsets of users
- Design experiments with clear success metrics defined before the test begins
- Ensure statistically significant sample sizes and test durations to produce reliable results
- Start with low-risk UI changes in the legacy system before testing more fundamental workflow modifications
- Instrument the legacy application with analytics to capture user behavior data needed for comparison
- Create a process for analyzing results and making data-driven decisions about which variant to adopt

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Replaces subjective design debates with data-driven decisions
- Reduces the risk of rolling out changes that degrade user experience
- Provides measurable evidence of improvement to justify modernization investments
- Enables incremental improvement of legacy UIs without full redesign

**Costs and Risks:**
- Legacy systems often lack the instrumentation needed for proper experiment tracking
- Adding feature flagging to legacy code increases complexity and requires careful cleanup
- Poorly designed experiments can produce misleading results that drive wrong decisions
- Running multiple concurrent experiments can produce interaction effects that confound results
- Some changes in legacy systems are too deeply embedded to be easily toggled between variants

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An insurance company wanted to modernize the claims submission workflow in their legacy web portal but could not agree internally on the best approach. The team implemented a feature flag system and created two alternative workflows alongside the existing one, routing 33% of users to each version. After four weeks, the data showed that the simplified three-step workflow had a 28% higher completion rate and 40% fewer support tickets compared to the original seven-step process. This evidence resolved months of internal debate and provided concrete justification for the modernization investment.
