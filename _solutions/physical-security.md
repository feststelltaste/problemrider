---
title: Physical Security
description: Access and entry protection for IT infrastructure through structural and organizational measures
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/physical-security
problems:
- data-protection-risk
- system-outages
- regulatory-compliance-drift
- poor-system-environment
- monitoring-gaps
- secret-management-problems
layout: solution
---

## How to Apply ◆

> Legacy systems often run on physical hardware in server rooms with outdated or inadequate physical access controls. Physical security protects IT infrastructure from unauthorized physical access, theft, tampering, and environmental hazards.

- Audit physical access to all locations where legacy system hardware resides: server rooms, network closets, tape storage areas, and any workstations with direct access to legacy system interfaces. Identify who currently has access and whether that access is justified.
- Implement access control mechanisms (badge readers, biometric scanners, locked cabinets) for all areas containing legacy system hardware. Replace shared keys and door codes with individually tracked access credentials.
- Deploy environmental monitoring for server rooms: temperature sensors, humidity sensors, water leak detection, and smoke detectors. Legacy hardware may have tighter environmental tolerances than modern equipment.
- Implement video surveillance and access logging for sensitive areas. Physical access logs should be retained and reviewed periodically, and access events should correlate with authorized work orders.
- Secure removable media and portable storage. Legacy systems often use USB drives, tapes, or removable disks for data transfer and backup — these must be encrypted, tracked, and stored securely when not in use.
- Implement visitor management procedures for areas containing legacy system infrastructure: escort requirements, temporary access badges, and sign-in/sign-out logs.
- Plan for physical security of legacy hardware during moves, decommissioning, and disposal. Hard drives, tapes, and other media containing sensitive data must be securely wiped or destroyed when hardware is retired.

## Tradeoffs ⇄

> Physical security prevents unauthorized physical access to infrastructure, protecting against threats that logical controls cannot address, but it requires investment in facilities, equipment, and ongoing operational procedures.

**Benefits:**

- Prevents data theft, hardware tampering, and unauthorized access that bypass all logical security controls — physical access to hardware defeats most software protections.
- Protects against environmental threats (fire, flood, power failure) that can destroy legacy hardware and the data it contains.
- Supports compliance with security standards (ISO 27001, PCI DSS, HIPAA) that require documented physical access controls.
- Provides audit trail of physical access for investigation and compliance purposes.

**Costs and Risks:**

- Physical security improvements (access control systems, surveillance, environmental monitoring) require capital investment in facilities infrastructure.
- Overly restrictive physical access can delay legitimate maintenance and emergency response activities for legacy systems that require hands-on intervention.
- Legacy systems in remote locations (branch offices, factory floors, customer sites) may be difficult to secure physically to the same standard as a data center.
- Decommissioning legacy hardware with sensitive data requires secure destruction procedures that add cost and complexity.

## Examples

> The following scenarios illustrate how physical security protects legacy system infrastructure.

A legacy database server containing 10 years of customer financial records sits in a server room secured only by a combination lock. The combination has not been changed in 5 years and is known to at least 30 current and former employees. During an off-hours break-in, a hard drive is removed from the server, and the theft is not discovered until the next business day when the system fails to boot. The team implements badge-based access control with individual access logs, adds 24/7 video surveillance with 90-day retention, deploys tamper-detection sensors on server chassis, and enables full-disk encryption so that stolen drives are unreadable without the encryption key. Access is restricted to 5 authorized personnel, and access reviews are conducted monthly. The combination lock is replaced with a badge reader that generates an auditable access log for every entry.

A company operates legacy manufacturing control systems on workstations distributed across a factory floor. These workstations have USB ports enabled for data transfer and are physically accessible to all factory personnel. A security assessment reveals that anyone can insert a USB device containing malware or copy data from the manufacturing system. The team implements USB port locks on all legacy workstations, installs locking enclosures that prevent access to the computer chassis, and deploys a dedicated kiosk for approved data transfers with malware scanning. Physical access audits conducted quarterly verify that the controls remain in place and that no unauthorized modifications have been made to the legacy workstations.
