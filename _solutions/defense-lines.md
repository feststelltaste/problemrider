---
title: Defense Lines
description: Implementing security mechanisms in multiple layers and levels
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/security/defense-lines
problems:
- authentication-bypass-vulnerabilities
- authorization-flaws
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- data-protection-risk
- insecure-data-transmission
- cascade-failures
layout: solution
---

## How to Apply ◆

> Legacy systems typically rely on a single security control (often a firewall or login check) as their sole defense. Defense in depth implements multiple independent layers of security so that a failure in any single control does not result in a complete compromise.

- Map the existing security controls in the legacy system and identify layers where no controls exist. Common gaps include: network layer (no segmentation), transport layer (no encryption), application layer (no input validation), data layer (no encryption at rest), and monitoring layer (no intrusion detection).
- Implement security controls at each architectural layer independently, so that each layer provides protection even if adjacent layers are compromised: perimeter firewalls, network segmentation, TLS encryption, application-level authentication and authorization, input validation, database-level access controls, and data encryption at rest.
- Apply the principle that no single security control should be the sole defense for any critical asset. For example, protecting against SQL injection requires input validation at the application boundary, parameterized queries at the data access layer, and least-privilege database accounts that limit damage even if injection succeeds.
- Implement monitoring and alerting as a defensive layer: even if preventive controls fail, detection controls (intrusion detection systems, anomaly detection, audit logging) can identify and limit the impact of a breach.
- Add rate limiting and throttling as a defensive layer to slow down automated attacks, giving other layers time to detect and respond.
- Segment the legacy system into security zones with different trust levels. Communication between zones should pass through security gateways that enforce zone-specific policies.
- Implement fail-secure defaults: when a security control fails or is bypassed, the system should deny access rather than allowing it. Legacy systems often fail open, granting access when security checks cannot be performed.

## Tradeoffs ⇄

> Defense in depth ensures that no single point of failure leads to a complete security breach, but it increases system complexity and can impact performance.

**Benefits:**

- Prevents a single vulnerability or misconfiguration from compromising the entire system by ensuring multiple independent controls must be defeated.
- Provides time for detection and response even when preventive controls fail, limiting the impact of successful attacks.
- Accommodates the reality that legacy systems will always have some vulnerabilities by ensuring that those vulnerabilities are not directly exploitable.
- Allows incremental security improvement by adding defensive layers without requiring a complete security redesign.

**Costs and Risks:**

- Multiple security layers add complexity to the system architecture, making it harder to understand, configure, and troubleshoot.
- Each security layer adds latency and computational overhead, which accumulates across multiple layers.
- Overlapping controls can create a false sense of security if each layer assumes another layer handles a particular threat.
- Managing and maintaining multiple security layers requires more operational effort and expertise than a single control.

## How It Could Be

> The following scenarios illustrate how defense in depth prevents complete compromise in legacy systems.

A legacy web application is protected solely by a web application firewall (WAF) that filters malicious input. When an attacker discovers a WAF bypass technique using chunked transfer encoding, they successfully inject SQL into the application and extract the entire customer database. After the incident, the team implements defense in depth: the WAF remains as the first layer but is supplemented by parameterized queries in the application code (eliminating SQL injection at the source), a least-privilege database account that can only access specific tables, database-level column encryption for sensitive fields (so even if data is extracted, it remains encrypted), and a database activity monitor that alerts on unusual query patterns. When a subsequent attacker finds another WAF bypass, the parameterized queries block the injection. The team is notified of the attempt through the monitoring layer and patches the WAF rule, preventing future bypass attempts.

A legacy financial system relies on network perimeter security (firewall) as its primary defense, assuming that everything inside the corporate network is trusted. When an employee's laptop is compromised through a phishing attack, the attacker gains unrestricted access to all internal systems, including the financial system's administrative interface. The team implements network segmentation to isolate the financial system in its own security zone, adds mutual TLS for all connections to the financial system, requires multi-factor authentication for administrative access, implements application-level authorization that verifies each request against a permission model, and deploys an intrusion detection system that monitors for anomalous access patterns. When a subsequent phishing compromise occurs, the attacker cannot access the financial system from the compromised workstation because it is in a different network segment, and attempts to reach the financial system's segment trigger an intrusion detection alert.
