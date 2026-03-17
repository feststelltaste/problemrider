---
title: Single Points of Failure
description: Progress is blocked when specific knowledge holders or system components
  are unavailable, creating critical dependencies.
category:
- Management
- Process
- Team
related_problems:
- slug: maintenance-bottlenecks
  similarity: 0.65
- slug: cascade-failures
  similarity: 0.65
- slug: knowledge-dependency
  similarity: 0.6
- slug: bottleneck-formation
  similarity: 0.6
- slug: knowledge-silos
  similarity: 0.6
- slug: cascade-delays
  similarity: 0.6
layout: problem
---

## Description

Single points of failure occur when critical system knowledge, capabilities, or processes depend entirely on individual team members or specific system components. When these individuals are unavailable or when key components fail, entire projects can be blocked, critical issues cannot be resolved, and development progress stops. This creates significant organizational risk and reduces team resilience, making the organization vulnerable to disruption from personnel changes or system failures.

## Indicators ⟡

- Specific team members are essential for certain types of work
- Development stops when key individuals are unavailable
- Critical system components have no backup or redundancy
- Certain problems can only be solved by one person
- Team panics when key personnel are sick or on vacation

## Symptoms ▲

- [Maintenance Bottlenecks](maintenance-bottlenecks.md)
<br/>  When only one person or component can handle certain tasks, maintenance queues up and creates bottlenecks.
- [Slow Incident Resolution](slow-incident-resolution.md)
<br/>  When the sole expert is unavailable, incidents take much longer to resolve because no one else has the required knowledge.
- [Slow Development Velocity](slow-development-velocity.md)
<br/>  Development stalls when key individuals are unavailable, reducing overall team throughput.
- [Cascade Delays](cascade-delays.md)
<br/>  When a single point of failure becomes unavailable, dependent work items cascade into delays across multiple teams and projects.
- [Staff Availability Issues](staff-availability-issues.md)
<br/>  When critical work depends on specific individuals, their unavailability creates effective staffing gaps even when the team is otherwise fully staffed.
## Causes ▼

- [Knowledge Silos](knowledge-silos.md)
<br/>  When knowledge is concentrated in individuals rather than shared across the team, those individuals become single points of failure.
- [Knowledge Dependency](knowledge-dependency.md)
<br/>  Critical system knowledge residing with specific individuals creates dependencies that make them irreplaceable.
- [Information Decay](poor-documentation.md)
<br/>  Without documentation, knowledge stays locked in individuals' heads, making the organization dependent on their availability.
- [Skill Development Gaps](skill-development-gaps.md)
<br/>  When team members fail to develop breadth of skills, expertise concentrates in few individuals who become single points of failure.
## Detection Methods ○

- **Bus Factor Analysis:** Identify what would happen if key individuals were unavailable
- **Dependency Mapping:** Chart which work depends on specific people or systems
- **Knowledge Distribution Assessment:** Evaluate how evenly critical knowledge is distributed
- **Availability Impact Tracking:** Monitor how often individual unavailability blocks work
- **Cross-Training Audit:** Assess how many people can perform critical tasks

## Examples

The entire deployment process depends on one senior developer who knows the complex sequence of manual steps, server configurations, and troubleshooting procedures. When they're sick for a week, releases are completely blocked because nobody else understands how to safely deploy the application or fix deployment issues. The team discovers they have no documentation of the deployment process and that attempts by others to deploy result in system failures. Another example involves a legacy database system where only one team member understands the complex data migration scripts and performance tuning procedures. When they leave the company, the team faces a crisis because critical database maintenance tasks can no longer be performed, and new features requiring database changes are blocked indefinitely.
