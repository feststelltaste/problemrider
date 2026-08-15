---
title: Multi-Cloud Infrastructure as Code
description: Provisioning infrastructure declaratively with provider-agnostic modules
  for multiple clouds
category:
- Operations
- Architecture
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
related_solutions:
- slug: infrastructure-as-code
  similarity: 0.7
- slug: immutable-infrastructure
  similarity: 0.7
- slug: containerization
  similarity: 0.65
- slug: cloud-native-development
  similarity: 0.65
- slug: virtual-networks
  similarity: 0.65
- slug: standardized-deployment-scripts
  similarity: 0.65
---

## Description

Multi-cloud infrastructure as code provisions infrastructure declaratively through provider-agnostic modules — typically built with tools like Terraform or Pulumi — that expose a uniform interface while abstracting away the provider-specific resource definitions underneath, so the same module can target AWS, Azure, or another cloud with only its variables changed. This works by codifying what was previously manual console configuration and ad hoc shell scripts into version-controlled, reviewable definitions, starting from the simplest environment and validating that it reproduces the existing manual setup before extending the approach further. Legacy systems are frequently locked into a single cloud provider not because that provider was deliberately chosen for technical reasons but because the infrastructure was built up incrementally through manual clicks and provider-specific scripts over years, with nobody documenting the resulting topology in a form that could be reproduced elsewhere. This kind of accidental lock-in leaves an organization with no leverage in vendor pricing negotiations and no practical disaster recovery option if the primary provider has an outage or a contractual dispute, since redeploying the system anywhere else would mean reconstructing its infrastructure from scratch. Adopting provider-agnostic IaC changes vendor lock-in from an unavoidable structural condition into a deliberate, revisitable choice, though the abstraction that makes this portability possible also means giving up some provider-specific optimizations and advanced managed services that do not have an equivalent on other clouds.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A healthcare SaaS provider was locked into a single cloud vendor whose pricing had increased 40% over two years. Their infrastructure was provisioned through a mix of console clicks and shell scripts, making migration seem impossible. The team adopted Terraform with provider-agnostic modules, starting by codifying their staging environment. Over six months they created modules for compute, networking, storage, and database resources that could target either AWS or Azure. When contract renegotiation stalled, they demonstrated the ability to provision their full stack on the alternative cloud within hours, which gave them significant leverage and ultimately resulted in better pricing terms.
