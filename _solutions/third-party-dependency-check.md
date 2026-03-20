---
title: Third-Party Dependency Check
description: Regularly review dependencies on external software
category:
- Security
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/security/third-party-dependency-check
problems:
- dependency-version-conflicts
- obsolete-technologies
- vendor-dependency
- shared-dependencies
- technology-lock-in
- high-technical-debt
- breaking-changes
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Implement automated dependency scanning tools (e.g., OWASP Dependency-Check, Snyk, Dependabot) in the CI/CD pipeline
- Maintain an inventory of all third-party dependencies including version, license, and maintenance status
- Establish a policy for maximum acceptable vulnerability age and severity before mandatory updates
- Schedule regular dependency review sessions to assess the health and security posture of critical dependencies
- Create upgrade paths for dependencies that have reached end-of-life or are no longer receiving security patches
- Monitor dependency projects for signs of abandonment, ownership changes, or compromised releases
- Test dependency updates in isolated environments before rolling them out to production

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Identifies known vulnerabilities in dependencies before they are exploited
- Provides visibility into the security and maintenance health of the dependency portfolio
- Enables proactive planning for dependency migrations before they become urgent
- Reduces the risk of using abandoned or compromised libraries

**Costs and Risks:**
- Legacy systems often have deeply embedded dependencies that are difficult to update without breaking changes
- Automated scanners may not cover all dependency types, especially custom or vendored libraries
- Frequent dependency updates can introduce regressions if not properly tested
- Updating one dependency in a legacy system can trigger cascading version conflicts

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A media company's legacy content management system used a JSON parsing library that had been abandoned by its maintainer three years earlier. An automated dependency check flagged this library as having two unpatched high-severity vulnerabilities. Because the team had been conducting regular dependency reviews, they had already identified a migration path to a supported alternative and completed the switch within one sprint. Without the automated check, the vulnerabilities would have remained undetected, as the library functioned correctly and no one on the team monitored its security status manually.
