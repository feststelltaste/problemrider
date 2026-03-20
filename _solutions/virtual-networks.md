---
title: Virtual Networks
description: Abstracting network configurations through virtual networks
category:
- Operations
- Architecture
quality_tactics_url: https://qualitytactics.de/en/portability/virtual-networks
problems:
- technology-lock-in
- vendor-lock-in
- deployment-environment-inconsistencies
- configuration-drift
- poor-system-environment
- network-latency
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Map the existing physical network topology and identify dependencies on specific IP ranges, VLANs, or hardware appliances
- Introduce software-defined networking (SDN) or cloud virtual private clouds to decouple the application from physical network infrastructure
- Use overlay networks (e.g., Docker networks, Kubernetes network policies, VXLAN) to create portable network configurations
- Replace hardcoded IP addresses with DNS-based service discovery so applications resolve endpoints dynamically
- Define network policies as code using tools like Terraform or Calico to ensure consistency across environments
- Test network configurations in isolated virtual environments before applying them to production

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables consistent network configuration across development, staging, and production environments
- Simplifies migration between data centers or cloud providers since network topology is abstracted
- Allows rapid provisioning of isolated test environments with production-like networking
- Reduces dependency on physical network hardware and specific vendor configurations

**Costs and Risks:**
- Virtual network overlays add latency and complexity compared to direct physical networking
- Troubleshooting network issues becomes harder with additional abstraction layers
- Teams need new skills in software-defined networking and cloud networking concepts
- Some legacy applications assume specific network topologies that are difficult to virtualize

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A healthcare company operated a legacy system spread across three physical data centers with network configurations manually managed by a small infrastructure team. Moving to the cloud was blocked because the application relied on specific IP address ranges and VLAN configurations hardcoded into dozens of configuration files. The team introduced Kubernetes with Calico network policies, replacing IP-based addressing with DNS service discovery. Network policies were defined as code and applied consistently across all environments. The migration to cloud infrastructure was completed without changing application code, and new test environments could be provisioned with identical network configurations in minutes rather than weeks.
