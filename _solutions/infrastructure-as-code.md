---
title: Infrastructure as Code
description: Defining and managing infrastructure through code
category:
- Operations
quality_tactics_url: https://qualitytactics.de/en/maintainability/infrastructure-as-code/
problems:
- configuration-drift
- configuration-chaos
- deployment-environment-inconsistencies
- environment-variable-issues
- inadequate-configuration-management
- legacy-configuration-management-chaos
- manual-deployment-processes
- complex-deployment-process
- deployment-risk
- poor-system-environment
- poor-operational-concept
- operational-overhead
layout: solution
---

## How to Apply ◆

> Applying Infrastructure as Code to a legacy system means converting years of accumulated manual configuration and tribal knowledge into version-controlled, auditable definitions — starting with the most volatile and least understood parts of the infrastructure.

- Begin with an infrastructure audit: map what actually exists across production, staging, and development environments. In legacy systems, these environments have typically drifted far apart. Document the differences before writing any code — the gaps reveal where the highest risks are.
- Adopt an incremental import strategy rather than a big-bang re-creation. Use tools like Terraform's `import` command to bring existing manually provisioned resources under IaC management one resource group at a time, starting with the infrastructure that changes most frequently and causes the most incidents.
- Prioritize environment parity. Legacy systems typically have a "works in production but not in staging" problem rooted in configuration drift. Once the most critical infrastructure is codified, use the same IaC definitions for all environments and eliminate manual environment-specific overrides.
- Store all IaC definitions in version control alongside application code. For legacy systems with no prior version control discipline for infrastructure, even a basic Git repository for Terraform files is a transformative improvement over shared spreadsheets and runbook wikis.
- Enforce peer review for all infrastructure changes through pull requests, including a plan review step. The `terraform plan` output must be reviewed before any `apply` runs. This single change catches the class of accidental deletions and misconfigurations that legacy systems routinely suffer from.
- Capture the "why" in commit messages and comments. Legacy infrastructure often contains firewall rules, instance size choices, and network configurations whose original rationale is unknown. When codifying existing infrastructure, include explanatory comments that record what is known about the reasoning — even if that is only "this was already here and we don't know why."
- Separate infrastructure state files by blast radius. Do not put the entire legacy system's infrastructure in one state file. Separate networking, compute, database, and application tiers so that a mistake in one area cannot trigger destruction in another.
- Add static analysis and security scanning (tools like `tflint`, `checkov`) to the CI pipeline from the start. Legacy infrastructure often harbors security misconfigurations that have persisted for years; automated scanning makes these visible without requiring a manual security audit.

## Tradeoffs ⇄

> Infrastructure as Code transforms legacy environments from invisible and undocumented to auditable and reproducible, but the migration path carries real risk that must be managed carefully.

**Benefits:**

- Configuration drift — the defining operational problem of legacy systems — is eliminated or made visible. Environments converge because they are all derived from the same source definitions.
- Infrastructure changes gain a full audit trail through version control history: who changed what, when, and why. This is essential for compliance in regulated legacy environments where change records are required but have historically been maintained manually.
- Disaster recovery becomes credible. Legacy systems frequently lack tested recovery procedures; IaC provides the ability to reconstruct entire environments from code rather than from institutional memory and forensic investigation.
- Infrastructure knowledge escapes the heads of the few people who manually provisioned the current environment. The IaC definitions serve as executable documentation that any team member can read and run.
- Reusable modules enable consistency across multiple legacy system environments and reduce the risk of introducing new configuration errors when spinning up additional environments for testing or migration purposes.

**Costs and Risks:**

- The initial migration of manually provisioned legacy infrastructure into IaC is high-effort and high-risk. Importing existing resources into state is painstaking, and the risk of accidentally destroying running production infrastructure during the migration is real. Teams need controlled rollout plans and rollback procedures for the migration itself.
- Legacy infrastructure often contains resources whose ownership and purpose are unclear. Codifying unknown resources risks breaking undocumented dependencies; not codifying them leaves gaps in coverage. This ambiguity slows adoption.
- State file management introduces a new category of operational risk. A corrupted or lost Terraform state file for a legacy system's production infrastructure can make the infrastructure effectively unmanageable until the state is reconstructed — potentially more disruptive than the problems IaC was meant to solve.
- Legacy teams with limited IaC experience face a significant learning curve. Tools like Terraform have their own language, state model, and failure modes. In teams already stretched by maintenance work, training takes time that is not always available.
- Compliance and change management processes in legacy organizations may require formal approval workflows that do not map cleanly onto pull-request-based IaC practices. Reconciling IaC velocity with change advisory board requirements is a common friction point.

## Examples

> The organizations that gain the most from Infrastructure as Code on legacy systems are those where years of manual provisioning have produced environments that nobody fully understands.

A retail company operating a decade-old e-commerce platform had accumulated several hundred EC2 instances, dozens of security groups, and hundreds of database parameter configurations — all provisioned manually over the years by a rotating cast of operations engineers. When a key engineer left, the team realized they could not confidently answer basic questions about their own infrastructure: which instances served which functions, which security group rules were still needed, and why certain instances used specific instance types. They began a systematic IaC migration, importing resources into Terraform state and documenting their purpose in the process. The audit that the migration forced upon them revealed seventeen instances that had been running for over two years with no clear ownership or purpose, and fourteen overly permissive security group rules that had been added during past incidents and never tightened.

A public sector organization running a legacy case management system had environments — production, pre-production, test, and developer — that had diverged so significantly over eight years that a bug fix verified in test regularly behaved differently in production. The team used Terraform to codify production as the authoritative baseline, then rebuilt all other environments from the same definitions with environment-specific parameters. Within three months, the "works in test, fails in production" incident category had dropped by roughly sixty percent, and the team had identified several production-specific configurations that should have been applied to test environments years earlier but never were.

A financial services firm needed to meet new regulatory requirements for infrastructure change auditing. Their legacy approach — manual changes applied by individual engineers through cloud console access — produced no usable change trail beyond raw cloud provider audit logs that were expensive to query and difficult to interpret. Migrating to IaC with mandatory pull request review addressed the audit requirement directly: every change was now a reviewed, approved, version-controlled event with a commit message explaining the business rationale. The audit evidence for regulatory review became a simple export of the Git log for the infrastructure repository.
