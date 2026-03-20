---
title: Honeypots
description: Deploying specially secured systems as bait for attackers
category:
- Security
quality_tactics_url: https://qualitytactics.de/en/security/honeypots
problems:
- monitoring-gaps
- authentication-bypass-vulnerabilities
- slow-incident-resolution
- data-protection-risk
- insufficient-audit-logging
layout: solution
---

## How to Apply ◆

> Legacy systems are attractive targets for attackers due to their known vulnerabilities and often weak monitoring. Honeypots complement existing security controls by deploying decoy systems that attract and detect attackers, providing early warning and intelligence about attack methods.

- Deploy low-interaction honeypots that emulate the legacy system's external interfaces (login pages, API endpoints, database ports) but contain no real data. These are quick to set up and detect automated scanning and opportunistic attacks.
- Place honeypot endpoints within the legacy system's network segment to detect lateral movement. Internal honeypots that should never receive legitimate traffic provide high-confidence alerts — any connection to them indicates unauthorized activity or compromise.
- Create honeytoken credentials (fake database accounts, API keys, service credentials) embedded in locations attackers commonly search: configuration files, source code repositories, and shared network drives. Any use of these credentials triggers an immediate alert.
- Deploy honeypot files (fake customer databases, dummy configuration files with attractive names) in shared storage locations. Access to these files, which no legitimate user or process should touch, indicates either an insider threat or an attacker with system access.
- Configure detailed logging on all honeypot components to capture attacker techniques, tools, and objectives. This intelligence improves defenses on the real legacy system by revealing what attackers are targeting.
- Ensure honeypots are sufficiently isolated so that an attacker who compromises the honeypot cannot pivot to real systems. Honeypots should appear connected to the network but be contained within a monitored sandbox.

## Tradeoffs ⇄

> Honeypots provide early attack detection and threat intelligence with low false-positive rates, but they must be maintained realistically and properly isolated.

**Benefits:**

- Detects attacks that bypass other security controls by providing high-confidence alerts — any interaction with a honeypot is suspicious by definition.
- Provides intelligence about attacker tools, techniques, and objectives that improves defenses on the real legacy system.
- Diverts attacker attention and effort to worthless targets, buying time for detection and response.
- Low false-positive rate since no legitimate user or system should interact with honeypot resources.

**Costs and Risks:**

- A compromised honeypot that is not properly isolated can be used as a pivot point to attack real systems.
- Honeypots require ongoing maintenance to remain realistic — outdated or obviously fake honeypots are easily identified and ignored by sophisticated attackers.
- Deploying honeypots introduces additional systems that must be monitored, patched (or deliberately left unpatched in a controlled manner), and managed.
- Legal and ethical considerations may apply to recording attacker activity, depending on jurisdiction.

## How It Could Be

> The following scenarios illustrate how honeypots detect threats targeting legacy systems.

A company runs a legacy database server that has been targeted by SQL injection attacks. They deploy a honeypot that emulates the legacy database's login interface on a nearby IP address with an intentionally weak password. Within a week, the honeypot logs 14 connection attempts from three different IP addresses using automated credential brute-forcing tools. The captured attack patterns reveal that the attackers are using a specific exploit toolkit targeting the legacy database version. This intelligence enables the security team to update their intrusion detection signatures and firewall rules to block these specific attack patterns on the real database server, preventing attacks that would not have been detected by existing monitoring.

A legacy source code repository contains hardcoded database credentials in configuration files. Rather than immediately rotating these credentials (which would require coordinating changes across multiple legacy components), the security team creates honeytoken database credentials in the same configuration files alongside the real ones. The honeytokens are monitored — any authentication attempt using them triggers an alert. Three weeks after deployment, an alert fires: someone is attempting to authenticate to the database using the honeytoken credentials. Investigation reveals that a contractor's laptop was compromised, and the attacker extracted credentials from a cloned repository. The honeytoken alert provides 4 hours of warning before the attacker attempts the real credentials, which the team rotates in that window.
