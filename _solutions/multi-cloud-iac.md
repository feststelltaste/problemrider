---
title: Multi-Cloud Infrastructure as Code
description: Provisioning infrastructure declaratively with provider-agnostic modules for multiple clouds
category:
- Operations
- Architecture
quality_tactics_url: https://qualitytactics.de/en/portability/multi-cloud-iac
problems:
- vendor-lock-in
- vendor-dependency
- vendor-dependency-entrapment
- technology-lock-in
- configuration-drift
- complex-deployment-process
- deployment-environment-inconsistencies
- manual-deployment-processes
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Audit existing infrastructure provisioning scripts and manual runbooks to understand the current deployment topology
- Choose a provider-agnostic IaC tool such as Terraform or Pulumi that supports multiple cloud providers
- Abstract provider-specific resource definitions into reusable modules that expose a uniform interface
- Start by codifying the simplest environment (e.g., staging) and validate parity with the existing manual setup
- Use variables and workspaces to parameterize cloud-specific details while keeping the module structure identical
- Integrate IaC into CI/CD pipelines so infrastructure changes go through code review and automated validation
- Maintain a state management strategy with remote backends and state locking to prevent drift

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces vendor lock-in by making cloud provider switches a configuration change rather than a rewrite
- Ensures environment consistency through declarative, version-controlled infrastructure definitions
- Eliminates manual provisioning errors and configuration drift across environments
- Enables disaster recovery scenarios where workloads can be redeployed on an alternative cloud

**Costs and Risks:**
- Provider-agnostic abstractions may sacrifice cloud-specific optimizations and advanced features
- Maintaining multi-cloud modules adds complexity compared to single-provider templates
- State management across providers introduces additional operational burden
- Teams need training on IaC tooling and cloud-agnostic design patterns
- Not all services have equivalent offerings across cloud providers

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A healthcare SaaS provider was locked into a single cloud vendor whose pricing had increased 40% over two years. Their infrastructure was provisioned through a mix of console clicks and shell scripts, making migration seem impossible. The team adopted Terraform with provider-agnostic modules, starting by codifying their staging environment. Over six months they created modules for compute, networking, storage, and database resources that could target either AWS or Azure. When contract renegotiation stalled, they demonstrated the ability to provision their full stack on the alternative cloud within hours, which gave them significant leverage and ultimately resulted in better pricing terms.
