---
title: Network Segmentation
description: Divide the network into security zones with separate trust levels
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/security/network-segmentation
problems:
- cascade-failures
- insecure-data-transmission
- data-protection-risk
- system-outages
- authorization-flaws
- monitoring-gaps
- poor-system-environment
layout: solution
---

## How to Apply ◆

> Legacy systems often operate on flat networks where any compromised component can reach any other component. Network segmentation divides the network into zones with different trust levels, limiting lateral movement and containing the blast radius of compromises.

- Map all network communication paths between legacy system components and identify which connections are actually required for the system to function. Many legacy systems have open network access between components that do not need to communicate directly.
- Define security zones based on data sensitivity and trust level: DMZ for internet-facing components, application zone for business logic, data zone for databases, management zone for administrative access, and isolated zones for legacy components with known vulnerabilities.
- Implement firewall rules or network policies between zones that allow only the specific ports, protocols, and source/destination pairs required for legitimate communication. Default-deny rules ensure that any new, unapproved communication path is blocked.
- Place legacy systems with known, unpatched vulnerabilities in isolated segments with strictly limited inbound and outbound access. These segments should have enhanced monitoring to detect exploitation attempts.
- Implement micro-segmentation for critical components: individual database servers, payment processing systems, and administrative interfaces should have their own network policies even within a broader zone.
- Deploy network monitoring at zone boundaries to detect unauthorized communication attempts. Any traffic that hits a deny rule represents either a misconfiguration or an attack attempt and should generate an alert.
- Document the network architecture and segmentation rules so that infrastructure changes maintain the intended security boundaries rather than inadvertently creating new paths through zones.

## Tradeoffs ⇄

> Network segmentation contains the blast radius of compromises and limits lateral movement, but it adds network complexity and may impact legacy system communication patterns.

**Benefits:**

- Limits the impact of a security breach by confining the attacker to the compromised segment rather than giving access to the entire network.
- Provides compensating controls for legacy systems that cannot be patched by isolating them from potential attack vectors.
- Enables different security monitoring intensities for different zones based on their sensitivity and risk level.
- Supports regulatory compliance by demonstrating that sensitive data environments are isolated from general-purpose networks.

**Costs and Risks:**

- Legacy systems may have undocumented network dependencies that break when segmentation is implemented, requiring careful discovery and testing.
- Network segmentation adds operational complexity to infrastructure management, requiring careful change control to maintain segmentation rules.
- Performance may be impacted if segmentation introduces additional network hops or firewall processing for high-volume inter-component communication.
- Improperly implemented segmentation can create a false sense of security if exceptions and bypass rules are too permissive.

## How It Could Be

> The following scenarios illustrate how network segmentation protects legacy systems.

A legacy web application, its database server, and a file server all reside on the same flat network segment. An attacker exploits a vulnerability in the web application and immediately discovers that they can connect directly to the database server on its default port, bypassing the application's access controls entirely. After implementing network segmentation, the web application is placed in an application zone, the database in a data zone, and the file server in a storage zone. Firewall rules permit only the application server to connect to the database on the specific port used by the application, and only the application server can access the file server. When a subsequent web application vulnerability is exploited, the attacker finds they can no longer reach the database directly — the firewall blocks all non-permitted connections, and the attempt triggers a network monitoring alert that initiates incident response.

A company operates a legacy mainframe that processes financial transactions alongside modern web services on the same corporate network. A ransomware outbreak that enters through a phishing email on a corporate workstation spreads across the flat network, eventually reaching the mainframe's network interface. The mainframe's batch processing halts for 48 hours while the ransomware is contained. After recovery, the team implements network segmentation that places the mainframe in an isolated high-security zone. Only two specific application servers are permitted to communicate with the mainframe, and only on the specific ports used for transaction submission and result retrieval. All other network traffic to the mainframe zone is blocked and logged. When a subsequent ransomware incident affects the corporate network, the mainframe zone is completely unaffected because no communication path exists from the infected segment to the mainframe zone.
