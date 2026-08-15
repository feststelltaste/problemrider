---
title: Abuse Case Definition
description: Describing undesirable use cases from the perspective of attackers
category:
- Security
- Requirements
problems:
- authentication-bypass-vulnerabilities
- authorization-flaws
- cross-site-scripting-vulnerabilities
- sql-injection-vulnerabilities
- buffer-overflow-vulnerabilities
- quality-blind-spots
- inadequate-requirements-gathering
- data-protection-risk
layout: solution
related_solutions:
- slug: authentication
  similarity: 0.75
- slug: negative-testing
  similarity: 0.7
- slug: penetration-tests
  similarity: 0.7
- slug: authorization
  similarity: 0.7
- slug: logging-and-monitoring
  similarity: 0.7
- slug: honeypots
  similarity: 0.7
---

## Description

Abuse case definition is a requirements-analysis technique that systematically documents how a system's functionality could be misused or attacked, producing a companion abuse case for every legitimate use case that the system was originally designed around. Where a use case describes what an authorized user is trying to accomplish, its abuse case describes what an attacker could achieve by exploiting the same functionality, together with the actor, the attack vector, the preconditions, and the resulting impact. Legacy systems are especially exposed to this gap because their original requirements were typically written when the system operated behind a firewall or internal network, and security threats were neither anticipated nor documented as first-class requirements; over time, the system's actual exposure has grown while its requirements have not been revisited. Defining abuse cases retroactively turns implicit, unstated security assumptions into explicit and testable statements, giving the team a structured way to decide which threats matter most and to translate them directly into concrete security test cases. Because the exercise requires both deep knowledge of the legacy system's internals and current knowledge of attacker techniques, it works best as a collaborative workshop between developers and security specialists rather than a task either group can complete alone. The resulting abuse case catalog also serves as durable documentation that justifies subsequent security investment to stakeholders who might otherwise not see the risk in functionality that has "always worked fine."

## How to Apply ◆

> Legacy systems are typically built with only legitimate use cases in mind, leaving security considerations as an afterthought. Abuse case definition systematically identifies how attackers can misuse the system, transforming implicit security assumptions into explicit, testable requirements.

- For each existing use case in the legacy system, create a corresponding abuse case that describes how an attacker might exploit or misuse that functionality. For example, if the use case is "user logs in," the abuse case is "attacker brute-forces login credentials" or "attacker bypasses authentication via session fixation."
- Involve both developers and security specialists in abuse case workshops. Developers understand the system's internals and know where shortcuts were taken, while security specialists bring knowledge of common attack patterns and exploitation techniques.
- Prioritize abuse case analysis for the most security-sensitive parts of the legacy system: authentication, authorization, payment processing, personal data handling, and administrative functions.
- Document each abuse case with a threat actor profile (who would attempt this), the attack vector (how they would do it), the preconditions (what access or knowledge they need), and the impact (what damage would result). This structured format makes abuse cases actionable for both testing and remediation.
- Use the STRIDE framework (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege) as a checklist to ensure comprehensive coverage across different threat categories.
- Translate abuse cases into concrete security test cases that can be executed manually or automated as part of the testing pipeline. Each abuse case should produce at least one test that attempts the described attack and verifies the system defends against it.

## Tradeoffs ⇄

> Abuse case definition shifts security thinking from reactive patching to proactive threat identification, but it requires specialized knowledge and ongoing effort to remain relevant.

**Benefits:**

- Systematically identifies security gaps that conventional requirements analysis misses, particularly in legacy systems where security was not a design priority.
- Produces concrete, testable security requirements that translate directly into security test cases and acceptance criteria.
- Builds shared understanding of security threats across the development team, improving the overall security awareness and code quality.
- Provides documentation that justifies security investments to stakeholders by describing realistic attack scenarios and their potential impact.

**Costs and Risks:**

- Requires security expertise that may not exist on the team, necessitating external consultants or training.
- Abuse case analysis can produce an overwhelming number of scenarios, requiring careful prioritization to focus on the highest-risk items.
- Abuse cases become outdated as attack techniques evolve, requiring periodic review and update.
- Without corresponding security tests and remediation, abuse case documentation provides awareness but not protection.

## How It Could Be

> The following scenarios illustrate how abuse case definition uncovers security vulnerabilities in legacy systems.

A legacy healthcare portal allows patients to view their medical records through a web interface. The original requirements specified only the legitimate use case: "Patient views their own records." An abuse case workshop identifies several attacker scenarios: an authenticated patient manipulates URL parameters to access another patient's records (insecure direct object reference), an attacker intercepts unencrypted API responses containing medical data, and a disgruntled employee uses their administrative access to export patient records in bulk. Testing these abuse cases reveals that the system indeed allows record access by modifying the patient ID parameter in the URL, as the backend performs no authorization check beyond initial login. This finding leads to the implementation of row-level access control, fixing a vulnerability that had existed for eight years.

A legacy financial application processes wire transfers through a multi-step workflow. Abuse case analysis identifies that an attacker with access to a low-privilege "viewer" account could potentially manipulate the transfer approval process by replaying captured HTTP requests from an authorized approver. Testing confirms that the system validates the session cookie but does not verify that the authenticated user has approval authority for the specific transaction. The abuse case directly informs the remediation: adding role-based authorization checks at every step of the approval workflow, not just at the initial login.
