---
title: Shared Dependencies
description: A situation where multiple components or services share a common set
  of libraries and frameworks.
category:
- Architecture
- Operations
related_problems:
- slug: shared-database
  similarity: 0.75
- slug: deployment-coupling
  similarity: 0.65
- slug: hidden-dependencies
  similarity: 0.6
- slug: circular-dependency-problems
  similarity: 0.6
- slug: tight-coupling-issues
  similarity: 0.6
- slug: vendor-dependency
  similarity: 0.55
layout: problem
---

## Description
Shared dependencies is a situation where multiple components or services share a common set of libraries and frameworks. This is a common problem in monolithic architectures, where all the components are tightly coupled and deployed as a single unit. Shared dependencies can lead to a number of problems, including deployment coupling, technology lock-in, and dependency version conflicts.

## Indicators ⟡
- Multiple components or services are using the same libraries and frameworks.
- It is not possible to update a library or framework for one component or service without affecting the others.
- There are often dependency version conflicts between different components or services.
- The system is difficult to maintain and extend.

## Symptoms ▲

- [Deployment Coupling](deployment-coupling.md)
<br/>  Components sharing dependencies must be deployed together when shared libraries are updated, creating deployment coupling.
- [Dependency Version Conflicts](dependency-version-conflicts.md)
<br/>  Different components may need different versions of shared libraries, creating version conflicts that are difficult to resolve.
- [Technology Lock-In](technology-lock-in.md)
<br/>  Shared dependencies lock all consuming components to the same technology versions, making it impossible to upgrade one without upgrading all.
- [Ripple Effect of Changes](ripple-effect-of-changes.md)
<br/>  Updating a shared dependency can have unexpected effects across all components that consume it, creating widespread ripple effects.
- [Maintenance Bottlenecks](maintenance-bottlenecks.md)
<br/>  Changes to shared libraries require coordination across all consuming teams, creating bottlenecks in the maintenance process.
## Causes ▼

- [Monolithic Architecture Constraints](monolithic-architecture-constraints.md)
<br/>  Monolithic systems naturally share all dependencies in a single build, and this pattern carries over when components are partially separated.
- [Convenience-Driven Development](convenience-driven-development.md)
<br/>  Reusing existing shared libraries is the path of least resistance for new components, even when it creates problematic coupling.
- [Code Duplication](code-duplication.md)
<br/>  Fear of code duplication drives teams to share libraries rather than allowing controlled duplication that would provide independence.
## Detection Methods ○
- **Dependency Analysis Tools:** Use tools to analyze the dependencies of the system to identify which libraries and frameworks are being shared by multiple components or services.
- **Developer Surveys:** Ask developers if they feel like they are able to update the libraries and frameworks for their components or services without affecting others.
- **Build and Test Log Analysis:** Analyze the build and test logs to identify dependency version conflicts.

## Examples
A company has a large, monolithic e-commerce application. The application is composed of a number of different components, including a product catalog, a shopping cart, and a payment gateway. All of the components share a common set of libraries and frameworks. When the development team wants to update a library for the product catalog, they have to be careful not to break the shopping cart or the payment gateway. This makes it difficult to update the libraries, and it often leads to problems.
