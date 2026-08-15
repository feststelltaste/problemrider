---
title: Continuous Delivery
description: Deliver functionality frequently and incrementally
category:
- Process
- Operations
problems:
- long-release-cycles
- complex-deployment-process
- manual-deployment-processes
- deployment-risk
- large-risky-releases
- release-anxiety
- immature-delivery-strategy
- delayed-value-delivery
- extended-cycle-times
- increased-time-to-market
- uneven-work-flow
layout: solution
related_solutions:
- slug: continuous-integration-and-delivery
  similarity: 0.85
- slug: ci-cd-pipeline
  similarity: 0.85
- slug: continuous-deployment
  similarity: 0.8
- slug: feature-driven-development
  similarity: 0.75
- slug: rollback-mechanisms
  similarity: 0.75
- slug: trunk-based-development
  similarity: 0.75
---

## Description

Continuous delivery keeps the codebase in a state where it could be released to production at any time, by automating the build, test, and packaging pipeline so that every change produces a deployable artifact rather than accumulating in an unreleased branch until the next scheduled release. Legacy systems are often locked into infrequent, large "big bang" releases precisely because deployment is manual, error-prone, and dreaded, which creates a vicious cycle: infrequent releases bundle more changes, larger releases carry more risk, and higher risk reinforces the reluctance to release more often. Automating the pipeline and adopting practices like trunk-based development, feature flags, and automated smoke tests breaks this cycle by making each release small enough to reason about and safe enough to reverse quickly if something goes wrong. Because feature flags decouple deployment from feature activation, code for an incomplete feature can move through the pipeline and sit dormant in production without being exposed to users, which is particularly valuable when a legacy system's architecture makes long-lived feature branches costly to maintain. The main cost is upfront: building reliable automation and sufficient automated test coverage for a legacy build process that has never been fully automated before requires real investment and a cultural shift away from treating releases as rare, high-ceremony events.

## How to Apply ◆

- Automate the build, test, and deployment pipeline for the legacy system, starting with the most error-prone manual steps.
- Implement trunk-based development or short-lived feature branches to reduce merge complexity in the legacy codebase.
- Deploy to production frequently in small increments rather than large, risky releases.
- Use feature flags to decouple deployment from feature activation, allowing code to be deployed without exposing incomplete functionality.
- Build automated smoke tests that verify core legacy system functionality after each deployment.
- Create automated rollback capabilities to reduce the risk of deploying changes to legacy systems.
- Standardize environments using infrastructure as code to eliminate "works on my machine" problems.

## Tradeoffs ⇄

**Benefits:**
- Reduces deployment risk by making each release smaller and more predictable.
- Shortens feedback loops, allowing teams to detect and fix issues faster.
- Eliminates manual deployment errors that are common in legacy system releases.
- Enables incremental modernization by allowing small improvements to reach production quickly.

**Costs:**
- Requires significant upfront investment to automate legacy system builds and deployments.
- Legacy systems may have dependencies or architectural constraints that make frequent deployment difficult.
- Requires comprehensive automated testing to maintain confidence in frequent releases.
- Cultural shift from infrequent "big bang" releases requires team adaptation and management support.

## How It Could Be

A legacy content management system is deployed quarterly through a manual, two-day process involving multiple teams and handoff documents. Each release bundles months of changes, and rollbacks require restoring from backup. The team invests three months in building a CI/CD pipeline: automated builds, database migration scripts, environment provisioning, and smoke tests. They begin releasing weekly, then twice weekly. Deployment incidents drop dramatically because each release contains fewer changes, and the automated pipeline eliminates the human errors that plagued manual deployments. The team discovers and fixes bugs within days instead of accumulating them for months, and stakeholders gain confidence in the delivery process.
