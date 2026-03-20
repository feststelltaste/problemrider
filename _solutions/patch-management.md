---
title: Patch Management
description: Apply security updates and patches promptly
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/patch-management
problems:
- obsolete-technologies
- system-outages
- data-protection-risk
- regulatory-compliance-drift
- deployment-risk
- dependency-version-conflicts
- technology-lock-in
- fear-of-change
layout: solution
---

## How to Apply ◆

> Legacy systems are frequently left unpatched due to fear of breaking functionality, lack of testing infrastructure, or vendor abandonment. Patch management establishes systematic processes for evaluating, testing, and applying security updates.

- Inventory all software components in the legacy system (operating systems, application frameworks, libraries, databases, middleware) and track their current versions against known vulnerability databases (CVE, NVD).
- Establish a patch classification and prioritization scheme: critical vulnerabilities in internet-facing components need immediate attention, while lower-severity issues in isolated internal components can follow a regular schedule.
- Create a testing pipeline for patches: apply patches to a staging environment that mirrors production, run regression tests, and verify that the legacy application functions correctly before applying to production.
- Define patching windows and rollback procedures. For legacy systems that require downtime for patching, schedule regular maintenance windows. For each patch, document the rollback procedure in case the update causes issues.
- For legacy systems running on end-of-life operating systems or frameworks that no longer receive patches, implement compensating controls: network isolation, virtual patching through WAF rules, application whitelisting, and enhanced monitoring.
- Automate vulnerability scanning to continuously identify unpatched components and newly disclosed vulnerabilities. Schedule weekly or daily scans and integrate findings into the team's work queue.
- Track patch compliance metrics (time from disclosure to patch application, percentage of systems up to date) and report them to management to maintain organizational commitment to patching.

## Tradeoffs ⇄

> Prompt patch management closes known vulnerability windows and reduces the attack surface, but it requires testing infrastructure, maintenance windows, and careful risk management.

**Benefits:**

- Closes known vulnerability windows before attackers can exploit them, directly reducing the system's attack surface.
- Maintains vendor support eligibility by keeping software within supported version ranges.
- Supports compliance with security standards that mandate timely application of security patches.
- Reduces the accumulation of technical debt from version drift by maintaining a regular update cadence.

**Costs and Risks:**

- Patches can break legacy application functionality, particularly when the application depends on specific behaviors of the underlying platform that the patch changes.
- Legacy systems often lack the automated testing infrastructure needed to confidently verify that patches do not introduce regressions.
- Patching may require downtime, which is difficult to schedule for systems with high availability requirements.
- End-of-life software that no longer receives patches cannot be remediated through patching, requiring migration or compensating controls.

## Examples

> The following scenarios illustrate how patch management reduces risk in legacy systems.

A legacy web server running Apache 2.2 has not been updated in 3 years. A critical remote code execution vulnerability (CVE) is disclosed that affects this version. Without a patch management process, the vulnerability goes unnoticed until a penetration test discovers it 6 months later. After implementing patch management, the team subscribes to vulnerability feeds for all software components in their stack. When a new critical Apache vulnerability is disclosed, automated scanning flags their servers within 24 hours. The team applies the patch to the staging environment, runs automated regression tests, verifies the legacy application functions correctly, and deploys to production within 72 hours of disclosure — well before exploit code becomes widely available. The team also creates a plan to migrate from Apache 2.2 to a currently supported version to ensure ongoing patch availability.

A legacy enterprise application depends on a Java 7 runtime that reached end of life years ago. No security patches are available, and upgrading to Java 8+ requires significant application changes due to deprecated API usage. The team implements compensating controls while planning the Java upgrade: they deploy a virtual patching layer using the WAF that blocks exploitation of known Java 7 vulnerabilities, restrict the application server's network access to only required destinations, and implement application whitelisting to prevent unauthorized code execution. These compensating controls provide protection during the 8-month Java migration project, and vulnerability scanning confirms that the known Java 7 CVEs cannot be exploited through the network path despite the runtime remaining unpatched.
