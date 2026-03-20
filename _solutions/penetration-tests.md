---
title: Penetration Tests
description: Uncovering security vulnerabilities through simulated attacks
category:
- Security
- Testing
quality_tactics_url: https://qualitytactics.de/en/security/penetration-tests
problems:
- authentication-bypass-vulnerabilities
- authorization-flaws
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- data-protection-risk
- insufficient-testing
- quality-blind-spots
layout: solution
---

## How to Apply ◆

> Legacy systems often contain security vulnerabilities that automated scanners miss because they require understanding of business logic, chaining of multiple weaknesses, or exploitation techniques specific to the legacy technology stack. Penetration testing simulates real-world attacks to discover exploitable vulnerabilities.

- Define the scope of the penetration test based on the legacy system's risk profile: which components, interfaces, and network segments are in scope, what testing methods are permitted, and what constitutes a finding versus expected behavior.
- Conduct both authenticated and unauthenticated testing. Authenticated tests (with valid credentials at various privilege levels) reveal authorization bypass and privilege escalation issues that are invisible from outside the application.
- Focus testing on legacy-specific risk areas: default credentials on administrative interfaces, deprecated TLS/SSL configurations, injection vulnerabilities in legacy code that predates modern frameworks, and insecure deserialization in legacy APIs.
- Test business logic vulnerabilities that automated scanners cannot detect: price manipulation, workflow bypass, race conditions in multi-step transactions, and access control gaps between different user roles.
- Perform network-level testing to identify unnecessary open ports, misconfigured services, and network paths that should not exist. Legacy infrastructure often has accumulated services that were started for debugging or testing and never removed.
- Classify findings by severity (critical, high, medium, low) and exploitability (easily exploitable vs. requiring specific conditions). Prioritize remediation based on the combination of impact and exploitability.
- Schedule regular penetration tests (at least annually, and after significant changes) and track remediation of findings from previous tests. Verify that fixes actually address the vulnerability rather than merely obscuring it.

## Tradeoffs ⇄

> Penetration testing provides realistic assessment of exploitable vulnerabilities from an attacker's perspective, but it is point-in-time and resource-intensive.

**Benefits:**

- Discovers exploitable vulnerabilities that automated scanners and code reviews miss, particularly business logic flaws and multi-step attack chains.
- Provides an attacker's-eye view of the system's security posture, revealing which vulnerabilities are practically exploitable versus merely theoretical.
- Validates that security controls (firewalls, WAFs, authentication, authorization) actually work as intended under adversarial conditions.
- Produces prioritized remediation guidance based on real exploitability rather than theoretical severity ratings.

**Costs and Risks:**

- Penetration testing is expensive and resource-intensive, requiring skilled security professionals with knowledge of the legacy technology stack.
- Testing is point-in-time — new vulnerabilities introduced after the test are not detected until the next engagement.
- Aggressive testing techniques can cause service disruptions, data corruption, or denial of service in legacy systems that are not designed to handle adversarial input gracefully.
- Penetration test findings can be overwhelming for teams with limited security remediation capacity, requiring careful prioritization.

## Examples

> The following scenarios illustrate how penetration testing uncovers critical vulnerabilities in legacy systems.

A legacy healthcare application undergoes its first penetration test after 10 years in production. The testers discover that the application's session management uses predictable session IDs (sequential integers), allowing an attacker to hijack any active session by incrementing the session ID. They also find that the "forgot password" function sends the actual password in plaintext email (rather than a reset link), revealing that passwords are stored in reversible encryption rather than one-way hashing. Additionally, the administrative interface is accessible from the internet without any additional authentication beyond the same login used by regular users. Each finding is rated critical, and the remediation plan prioritizes them in order: restricting administrative access to the internal network (immediate), implementing cryptographic session IDs (1 week), and migrating password storage to bcrypt with a reset flow (1 month).

A legacy financial trading platform receives annual penetration tests. The most recent test reveals a race condition in the order placement API: by submitting two orders simultaneously for the same limited-quantity security, an attacker can purchase more units than are available because the inventory check and the inventory decrement are not atomic. The testers demonstrate that this can be exploited with simple concurrent HTTP requests, without any special tools. The finding leads to the implementation of database-level locking on inventory checks, a vulnerability that automated vulnerability scanners would never detect because it requires understanding of the business logic and the ability to test concurrent request handling.
