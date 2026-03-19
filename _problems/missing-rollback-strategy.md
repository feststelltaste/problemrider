---
title: Missing Rollback Strategy
description: There's no tested method to undo a deployment if things go wrong, increasing
  risk
category:
- Code
- Management
- Process
related_problems:
- slug: deployment-risk
  similarity: 0.9
- slug: manual-deployment-processes
  similarity: 0.65
- slug: complex-deployment-process
  similarity: 0.6
- slug: large-risky-releases
  similarity: 0.6
- slug: immature-delivery-strategy
  similarity: 0.6
- slug: deployment-coupling
  similarity: 0.6
layout: problem
---

## Description

Deployment Risk occurs when teams deploy systems without having a reliable, tested method to quickly revert to a previous working state when problems arise. This creates significant risk during deployments, as any issues discovered post-deployment can only be resolved by fixing forward, which may take considerable time and cause extended outages. The absence of rollback capabilities often leads to deployment anxiety, longer incident resolution times, and greater impact when deployments go wrong.

## Indicators ⟡

- Deployment procedures that only document forward deployment steps
- Database migration scripts without corresponding rollback scripts
- Infrastructure changes that are difficult or impossible to reverse
- Deployment anxiety and reluctance to deploy during business hours
- Incident response plans that assume fixing forward as the only option
- No testing of rollback procedures during deployment planning
- Configuration changes that overwrite previous settings without backup

## Symptoms ▲

- [Slow Incident Resolution](slow-incident-resolution.md)
<br/>  When deployments go wrong and cannot be quickly reverted, incidents last longer and have greater impact.
- [Large, Risky Releases](large-risky-releases.md)
<br/>  Deployment anxiety from missing rollback leads teams to batch changes into fewer, larger releases that are even riskier.
- [Release Anxiety](release-anxiety.md)
<br/>  Without a tested rollback strategy, every deployment becomes a high-stakes event since problems cannot be easily reve....
## Causes ▼

- [Manual Deployment Processes](manual-deployment-processes.md)
<br/>  Manual deployment workflows make it difficult to implement and test reliable rollback procedures.
- [Complex Deployment Process](complex-deployment-process.md)
<br/>  Overly complex deployment processes make it impractical to define and test rollback steps for every component.
- [Immature Delivery Strategy](immature-delivery-strategy.md)
<br/>  Organizations with immature delivery practices often lack the discipline to plan and test rollback strategies as part of deployment.
## Detection Methods ○

- Review deployment documentation for rollback procedure coverage
- Audit database migration scripts for presence of rollback/down migrations
- Test rollback procedures in staging environments as part of deployment planning
- Assess infrastructure provisioning tools for state management and rollback capabilities
- Survey deployment teams about confidence in rollback options
- Review incident response procedures for rollback vs. fix-forward decision trees
- Examine deployment tooling for built-in rollback functionality
- Analyze historical incident data for cases where rollback would have reduced impact

## Examples

An e-commerce platform deploys a new payment processing feature during a routine Friday evening release. The deployment includes database schema changes that add new columns and modify existing constraints. Two hours after deployment, customer reports start flooding in about failed payment processing that's blocking orders. The team discovers a critical bug in the new payment logic that affects all transactions. However, they realize they cannot roll back because the database migrations are irreversible - they added required columns that can't be safely removed without data loss. The team is forced to spend the entire weekend debugging and fixing the payment issue in production while the e-commerce site loses revenue from failed transactions. A proper rollback strategy with reversible database changes could have restored service within minutes instead of days.
