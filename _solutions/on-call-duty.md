---
title: On-Call Duty
description: Ensuring employees are available to quickly respond to incidents and
  issues
category:
- Process
- Operations
problems:
- slow-incident-resolution
- constant-firefighting
- system-outages
- knowledge-silos
- poorly-defined-responsibilities
- developer-frustration-and-burnout
- overworked-teams
- increased-stress-and-burnout
- mental-fatigue
- lack-of-ownership-and-accountability
layout: solution
related_solutions:
- slug: runbooks
  similarity: 0.75
- slug: incident-management
  similarity: 0.75
- slug: sustainable-pace-practices
  similarity: 0.7
- slug: clear-roles-and-ownership
  similarity: 0.7
- slug: security-incident-handling
  similarity: 0.7
- slug: cross-functional-skill-development
  similarity: 0.7
---

## Description

On-call duty is a formal rotation that assigns specific people the responsibility of responding to production incidents outside normal working hours, replacing the informal arrangement where the same one or two people who understand the legacy system get called every time something breaks. Establishing a rotation with clear escalation paths, documented runbooks, and defined response expectations distributes operational knowledge across the team instead of concentrating it in a few overburdened individuals. In legacy systems, where institutional knowledge is often thin and unevenly spread, a well-run on-call rotation forces that knowledge to be written down and shared as part of onboarding new rotation members, while also creating accountability for fixing the recurring issues that generate repeat pages rather than tolerating them indefinitely.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Establish a fair rotation schedule that distributes on-call burden across all team members
- Provide clear escalation paths and runbooks so on-call engineers can handle legacy system issues effectively
- Define response time expectations for each severity level and communicate them to stakeholders
- Equip on-call engineers with necessary access, tools, and documentation for legacy system troubleshooting
- Compensate on-call duty appropriately to maintain team morale and willingness to participate
- Conduct regular on-call handoffs that include context about recent changes and known issues
- Review on-call metrics (page frequency, after-hours pages, MTTR) and address sources of excessive toil

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Ensures rapid response to production incidents at all hours
- Distributes operational knowledge across the team rather than relying on a few experts
- Creates accountability for production quality among developers
- Provides a structured alternative to ad hoc firefighting

**Costs and Risks:**
- On-call duty causes stress and can contribute to burnout if not managed well
- Frequent pages disrupt personal time and affect work-life balance
- Teams with limited legacy system knowledge may struggle during on-call shifts
- Under-staffed on-call rotations concentrate burden on too few people

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A software company had relied on two senior engineers who knew the legacy system best to handle all production issues, regardless of time. Both were burning out and had become single points of failure for operational knowledge. By implementing a formal on-call rotation with comprehensive runbooks and a buddy system pairing junior and senior engineers, the team distributed incident response across eight people. On-call page volume was also reduced by 60% because the rotation motivated the team to fix recurring issues rather than repeatedly work around them.
