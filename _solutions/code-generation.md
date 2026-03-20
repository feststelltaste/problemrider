---
title: Code Generation
description: Automatic creation of code parts based on templates or metadata
category:
- Code
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/code-generation
problems:
- code-duplication
- copy-paste-programming
- inconsistent-codebase
- maintenance-overhead
- slow-feature-development
- increased-cost-of-development
- increased-risk-of-bugs
layout: solution
---

## How to Apply ◆

> In legacy systems, code generation reduces boilerplate duplication and enforces consistency by generating repetitive code from templates or metadata rather than writing it by hand.

- Identify repetitive patterns in the legacy codebase that follow a predictable structure — data access objects, API client stubs, serialization code, and configuration classes are common candidates.
- Choose generation tools appropriate for the legacy system's technology stack (code generators, template engines, annotation processors, or schema-driven generators like OpenAPI or Protocol Buffers).
- Generate code from a single source of truth (database schemas, API specifications, or configuration files) to ensure consistency across the generated artifacts.
- Keep generated code clearly separated from hand-written code through naming conventions, directory structure, or build tool configuration so that developers do not accidentally modify generated files.
- Include the generation step in the build pipeline so that generated code stays synchronized with its source metadata.
- Use code generation during legacy migration to produce consistent boilerplate for the new system based on legacy schema or interface definitions.

## Tradeoffs ⇄

> Code generation eliminates boilerplate maintenance but introduces dependencies on generation tools and templates that must be managed.

**Benefits:**

- Eliminates entire classes of copy-paste bugs by generating repetitive code consistently from a single template.
- Speeds up development of repetitive code structures, especially when migrating many similar components from a legacy system.
- Ensures consistency across generated artifacts — when the template changes, all generated code changes uniformly.
- Reduces the amount of code developers need to write and review, focusing their attention on business logic.

**Costs and Risks:**

- Generated code can be difficult to debug when problems arise in the generation process rather than the generated output.
- The generation templates and tooling become critical dependencies that require maintenance and expertise.
- Over-reliance on code generation can lead to generated code that does not fit well in all contexts, requiring workarounds.
- Developers may not understand the generated code well enough to debug issues or recognize when generation is producing suboptimal output.

## How It Could Be

> The following scenario shows how code generation accelerates legacy system migration.

A financial services company was migrating from a legacy system with 180 database tables to a new microservices architecture. Each table needed a corresponding repository class, DTO, mapper, and REST endpoint in the new system — approximately 900 boilerplate files. Rather than writing these by hand, the team built a code generator that read the legacy database schema and produced all four artifacts for each table. The generator completed in seconds what would have taken weeks of manual coding and ensured that naming conventions, error handling patterns, and mapping logic were perfectly consistent across all 180 entities. When the team later decided to change the error response format across all endpoints, they updated the template and regenerated all endpoint classes in a single step.
