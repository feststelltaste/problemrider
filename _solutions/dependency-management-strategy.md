---
title: Dependency Management
description: Systematize the management and updating of external dependencies
category:
- Dependencies
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/dependency-management/
problems:
- dependency-version-conflicts
- vendor-lock-in
- vendor-dependency-entrapment
- vendor-dependency
- technology-lock-in
- breaking-changes
- legacy-api-versioning-nightmare
- api-versioning-conflicts
- shared-dependencies
- dependency-on-supplier
- technology-stack-fragmentation
- obsolete-technologies
- premature-technology-introduction
- vendor-relationship-strain
layout: solution
---

## How to Apply ◆

> Legacy systems frequently accumulate years of unmanaged dependency drift — libraries pinned to ancient versions, no lock files, and no visibility into transitive vulnerabilities — making systematic dependency management both urgent and technically challenging to introduce.

- Start by generating a complete dependency inventory including transitive dependencies using `mvn dependency:tree`, `npm ls`, or equivalent; for many legacy systems this is the first time the full scope of the dependency surface becomes visible, and the result is often alarming.
- Introduce lock files (package-lock.json, Pipfile.lock, pom.xml with fixed versions) immediately and commit them to version control — this establishes a reproducible baseline from which all future changes can be made deliberately rather than accidentally.
- Run a vulnerability scanner (Dependabot, Snyk, OWASP Dependency-Check) against the current dependency tree before making any updates; the goal is to understand current risk exposure, not to fix everything at once — triage by severity and start with critical and high findings.
- Define a written update policy for the team: critical security patches within one week, minor version updates within a sprint, major version upgrades scheduled quarterly — legacy teams often have no policy at all and respond only when something breaks.
- Update dependencies one library at a time in small increments rather than attempting a bulk upgrade of everything accumulated over years; jumping three major versions of a framework simultaneously is a project, not a task.
- Establish vulnerability scanning in the CI pipeline as a quality gate that blocks merges containing critical or high-severity unpatched dependencies — this prevents the accumulation from recurring after the initial cleanup.
- Audit licenses across the dependency tree using tools like FOSSA or license-checker; legacy systems often contain GPL-licensed libraries in commercial products due to decisions made without legal review years earlier.
- Monitor the health of key dependencies — libraries with no commits in two years, abandoned maintainers, or declining community activity represent maintenance risk that should trigger a replacement search before a crisis forces the issue.

## Tradeoffs ⇄

> Systematic dependency management converts a chronic background risk into a manageable, visible, and auditable process, but catching up on years of neglect requires sustained investment that competes with feature delivery.

**Benefits:**

- Vulnerability scanning and automated update pull requests (Dependabot/Renovate) surface security risks that would otherwise remain hidden for years inside the transitive dependency tree — critical in legacy systems where Log4Shell-style vulnerabilities can lurk undetected.
- Lock files and deterministic version resolution eliminate the "works on my machine" problems that plague legacy teams who have been building without them, making CI builds reproducible and debugging dramatically easier.
- Regular small updates prevent the accumulation of upgrade debt that eventually forces painful multi-version jumps; incremental updates are manageable, while multi-year gaps create weeks-long migration projects.
- A software bill of materials (SBOM) generated from the managed dependency tree satisfies regulatory and enterprise supply-chain requirements that are increasingly mandatory for legacy systems in regulated industries.
- Retiring unmaintained dependencies before they become critical forces early architectural conversations about replacements, giving teams control over the timeline rather than reacting to end-of-life announcements.

**Costs and Risks:**

- Initial cleanup of a long-neglected dependency tree in a large legacy system can require weeks of effort to triage vulnerabilities, resolve conflicts, and verify that updated libraries do not change behavior in critical code paths.
- Automated update pull requests from Dependabot produce false positives and low-severity noise that can overwhelm teams if no triage policy is in place, leading to alert fatigue and ignored updates.
- Legacy systems built without automated test coverage make dependency updates risky — without a test suite, there is no reliable way to verify that an updated library has not changed behavior in a way that breaks the application.
- Major version upgrades of foundational frameworks (Spring Boot 2 to 3, Angular 9 to 17) in legacy systems can require changes across hundreds of files and constitute projects in their own right, not just dependency bumps.
- Dependency on external maintainers whose priorities do not align with the legacy system's needs is an irreducible risk; a popular library abandoning support for an older Java version can force an upgrade cascade that the team is not prepared for.

## Examples

> The following scenarios illustrate what dependency management looks like when applied to real legacy system conditions.

A Java enterprise application built in 2014 was still running Spring 4 and Hibernate 4 in 2023. The team had avoided updates because the last major update attempt had caused a three-week integration crisis. When a security audit revealed seventeen high-severity CVEs in the transitive dependency tree — several in frameworks the team did not even know the application depended on — the project finally received budget for remediation. The team introduced OWASP Dependency-Check into the CI pipeline, established a written policy for security patch timelines, and worked through the upgrade backlog one library at a time over four sprints. The Spring upgrade alone required a dedicated two-week effort, but the staged approach prevented the project from turning into another integration crisis.

A retail company operating a Node.js backend discovered after running `npm ls` for the first time that their application had 1,400 transitive dependencies, many of which were pinned to versions published before 2019. Enabling Dependabot created an immediate flood of 200+ pull requests. The team established a triage process: critical CVEs were assigned to a developer immediately, minor version bumps were batched weekly, and major version bumps were scheduled monthly. Within three months, the dependency tree was current and the weekly batch had shrunk to a manageable set of a dozen routine updates.

A government contractor inherited a Python data processing pipeline that had no requirements lock file — just a `requirements.txt` with unpinned version ranges. The pipeline had worked for years on a specific server where the installed package versions happened to be compatible, but a new developer attempting to run it locally found it would not start due to incompatible transitive dependencies. Introducing `pip-tools` to generate a `requirements.txt.lock`, committing it to the repository, and scanning the pinned versions with Snyk revealed two critical vulnerabilities in pinned transitive packages that had been silently present for over eighteen months.
