---
title: Continuous Delivery
description: Deliver functionality frequently and incrementally
category:
- Process
- Operations
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/continuous-delivery
problems:
- long-release-cycles
- complex-deployment-process
- manual-deployment-processes
- deployment-risk
- large-risky-releases
- release-anxiety
- immature-delivery-strategy
- delayed-value-delivery
layout: solution
---

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

## Examples

A legacy content management system is deployed quarterly through a manual, two-day process involving multiple teams and handoff documents. Each release bundles months of changes, and rollbacks require restoring from backup. The team invests three months in building a CI/CD pipeline: automated builds, database migration scripts, environment provisioning, and smoke tests. They begin releasing weekly, then twice weekly. Deployment incidents drop dramatically because each release contains fewer changes, and the automated pipeline eliminates the human errors that plagued manual deployments. The team discovers and fixes bugs within days instead of accumulating them for months, and stakeholders gain confidence in the delivery process.
