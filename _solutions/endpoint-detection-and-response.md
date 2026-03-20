---
title: Endpoint Detection and Response
description: Continuously monitoring endpoints for threats in real-time
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/endpoint-detection-and-response
problems:
- system-outages
- monitoring-gaps
- slow-incident-resolution
- data-protection-risk
- authentication-bypass-vulnerabilities
- constant-firefighting
layout: solution
---

## How to Apply ◆

> Legacy systems often run on endpoints (servers, workstations, terminals) with minimal or no threat detection, making it impossible to identify compromise until significant damage has occurred. Endpoint Detection and Response (EDR) provides continuous visibility into endpoint activity and enables rapid threat response.

- Deploy EDR agents on all servers and workstations that host or access the legacy system. Ensure agents are compatible with the legacy operating systems in use — some legacy systems run on older OS versions that require specific EDR agent versions.
- Configure baseline behavioral profiles for each endpoint to distinguish normal activity from anomalous behavior. Legacy systems often have predictable, repetitive usage patterns that make anomaly detection particularly effective.
- Enable process monitoring to detect unauthorized process execution, including scripts, unauthorized binaries, and processes that should not run on production servers.
- Implement file integrity monitoring for critical system files, application binaries, and configuration files. Legacy systems are attractive targets for persistent threats that modify system files to maintain access.
- Configure automated response actions for high-confidence threats: isolate the endpoint from the network, terminate malicious processes, and alert the security team. For lower-confidence detections, create alerts without automated response to avoid disrupting legitimate legacy system operations.
- Integrate EDR telemetry with centralized SIEM for correlation across endpoints and network activity, enabling detection of lateral movement and multi-stage attacks.

## Tradeoffs ⇄

> EDR provides real-time threat visibility and rapid response capability for legacy system endpoints, but it requires compatible agents and careful tuning to avoid false positives.

**Benefits:**

- Detects compromises in real-time rather than after-the-fact, enabling immediate containment and reducing the impact of security incidents.
- Provides detailed forensic data (process trees, file modifications, network connections) that supports incident investigation.
- Enables automated containment actions that limit damage even when security personnel are not immediately available.
- Compensates for the lack of application-level security controls in legacy systems by monitoring endpoint behavior.

**Costs and Risks:**

- EDR agents consume CPU and memory resources on legacy servers that may already be resource-constrained.
- Compatibility issues with legacy operating systems or applications may limit EDR functionality or cause system instability.
- High false-positive rates on legacy systems with unusual but legitimate behavior patterns can overwhelm the security team.
- Automated response actions (endpoint isolation) can cause unplanned outages if triggered incorrectly on critical legacy system servers.

## How It Could Be

> The following scenarios illustrate how EDR protects legacy system environments.

A legacy accounting system runs on a Windows Server 2012 instance that has not received security patches in over a year due to concerns about application compatibility. An attacker exploits a known remote code execution vulnerability and establishes a persistent backdoor. Without EDR, the compromise goes undetected for three months, during which the attacker exfiltrates financial data. After deploying EDR, the agent detects a similar compromise attempt on another legacy server within minutes — the EDR identifies an unusual PowerShell process spawned from the accounting application's service account, a behavior that has never occurred in the endpoint's baseline profile. The agent automatically isolates the endpoint and alerts the security team, who contain the incident within 30 minutes.

A legacy manufacturing control system runs on dedicated workstations connected to both the corporate network and the factory floor network. EDR deployment on these workstations reveals that one workstation is communicating with an unknown external IP address at regular intervals — a pattern consistent with command-and-control beaconing. Investigation reveals that a USB drive used for data transfer between systems was infected with malware that established persistence on the workstation. The EDR containment blocks the external communication immediately, and the forensic telemetry from the EDR agent provides the complete infection timeline, enabling the team to verify that no factory floor systems were compromised.
