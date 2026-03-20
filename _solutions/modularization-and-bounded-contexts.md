---
title: Modularization
description: Divide application into small, independent, and reusable components
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/maintainability/modularization/
problems:
- monolithic-architecture-constraints
- high-coupling-and-low-cohesion
- tight-coupling-issues
- circular-dependency-problems
- complex-domain-model
- poor-domain-model
- shared-database
- shared-dependencies
- tangled-cross-cutting-concerns
- difficult-code-reuse
- god-object-anti-pattern
- spaghetti-code
- hidden-dependencies
- system-integration-blindness
layout: solution
---

## How to Apply ◆

> Applying modularization to a legacy system means imposing explicit boundaries on a codebase that evolved without them, starting with the highest-pain areas and using enforcement tools to stop the boundaries from immediately dissolving again.

- Begin with a dependency analysis of the existing codebase using tools like Structure101, Lattix, or language-specific analysis tools to produce an actual map of inter-component dependencies — most teams are surprised to find the real coupling is worse than assumed.
- Identify two or three major business capabilities (order management, billing, inventory) as initial module boundaries; resist the urge to immediately model every subdomain, since premature fine-grained boundaries in a system you do not yet understand often draw lines in the wrong places.
- Introduce build-system-level module boundaries (Maven modules, Gradle subprojects, npm workspaces) rather than relying on packaging conventions alone; convention-based boundaries erode within weeks under deadline pressure, but compilation errors do not.
- Use architecture testing tools such as ArchUnit or Dependency-Cruiser to encode the boundary rules as automated tests that run in CI; this makes boundary violations as visible and urgent as failing unit tests.
- Tackle shared databases as part of the modularization effort by identifying which tables are accessed by which business capabilities and assigning ownership; tables accessed by more than one capability are the primary risk for hidden coupling and deserve explicit anti-corruption layers or event-based synchronization.
- Eliminate circular dependencies as they are discovered, not later — extract shared concepts into a dedicated utility or kernel module rather than allowing modules to depend on each other in both directions.
- Assign a named owner to each module so that decisions about its interface, internal structure, and technical debt have a responsible party; unowned modules in legacy systems are where the most severe decay concentrates.
- Accept that the first module boundaries will be wrong in some places; design them to be refined, document the rationale, and plan to revisit them after the first quarter of working within the new structure.

## Tradeoffs ⇄

> Modularization imposes short-term friction — planning, enforcement tooling, refactoring effort — in exchange for long-term ability to change parts of the system independently.

**Benefits:**

- Reduces the cognitive overhead of working in a large legacy codebase by allowing developers to focus on one module at a time without needing to understand the entire system.
- Enables teams to work independently on separate modules without constant merge conflicts, reducing the coordination cost that plagues monolithic legacy development.
- Makes selective modernization possible: a well-bounded module can be rewritten, replaced, or extracted as a service without requiring changes to the modules around it.
- Localizes the blast radius of a change so that a defect or regression in one module is less likely to surface as an unexpected failure in a completely different part of the system.
- Supports incremental test coverage improvement because individual modules can be tested in isolation once their dependencies are exposed through interfaces.

**Costs and Risks:**

- Identifying correct module boundaries in a system with years of accumulated coupling requires significant analysis effort upfront, and the analysis is often blocked by undocumented behavior that only certain people understand.
- Wrong boundaries — drawn around technical layers rather than business capabilities, or drawn too finely before the domain is understood — create friction that persists until expensive refactoring corrects them.
- Shared databases are the most common obstacle to real modularization in legacy systems; splitting database ownership is technically and organizationally challenging, and teams frequently stop at package-level modularization while the database remains fully shared.
- Boundary enforcement requires governance: without regular dependency checks and a culture that treats boundary violations as seriously as bugs, the boundaries erode faster than they were established.
- Performance overhead from interface-mediated communication between modules — particularly where the legacy system previously used direct in-process calls across what are now module boundaries — may require measurement and optimization.

## How It Could Be

> The following scenarios show how modularization has been applied in practice to bring structure to systems where none previously existed.

A telecommunications company operated a customer management system that had grown over fourteen years into a single Java web application with over 800,000 lines of code and no package structure that reflected the business domain. A dependency analysis revealed 6,000 inter-class dependencies, including dozens of cycles. The team used a three-month architecture initiative to identify five core business capabilities — customer identity, contract management, billing, service provisioning, and support — and reorganized classes into Maven modules aligned to those boundaries. ArchUnit rules enforced the new structure in CI. Within six months, the billing module team was able to introduce a new pricing model without coordinating with the provisioning or identity teams.

A health insurance company discovered during a modernization assessment that its claims processing system had three different interpretations of the word "member" depending on which part of the legacy code was executing. In enrollment logic, a member was a policyholder. In claims adjudication, a member was an individual covered under a policy. In reporting, a member was a unique combination of policy and benefit period. These overlapping concepts had caused subtle data inconsistencies for years. By applying bounded context analysis, the team drew explicit boundaries around enrollment, adjudication, and reporting and gave each its own model of "member" with translations between them at the integration points. This eliminated the inconsistencies and made each team's domain model internally coherent.

A financial services firm needed to extract its trade reporting capability from a legacy C++ monolith so it could be offered as a managed service to partner firms. The extraction was expected to take six months but the actual effort revealed that the reporting code was entangled with the trade booking engine through 200 direct function calls and four shared global data structures. The team spent four months first creating a module boundary within the monolith — replacing direct function calls with an interface, eliminating the shared globals through a data access layer — before any extraction to a separate service was possible. The modularization work was the prerequisite that made the architectural change achievable at all.
