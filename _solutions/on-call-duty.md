---
title: On-Call Duty
description: Ensuring employees are available to quickly respond to incidents and issues
category:
- Process
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/on-call-duty
problems:
- slow-incident-resolution
- constant-firefighting
- system-outages
- knowledge-silos
- poorly-defined-responsibilities
- developer-frustration-and-burnout
- overworked-teams
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A software company had relied on two senior engineers who knew the legacy system best to handle all production issues, regardless of time. Both were burning out and had become single points of failure for operational knowledge. By implementing a formal on-call rotation with comprehensive runbooks and a buddy system pairing junior and senior engineers, the team distributed incident response across eight people. On-call page volume was also reduced by 60% because the rotation motivated the team to fix recurring issues rather than repeatedly work around them.
