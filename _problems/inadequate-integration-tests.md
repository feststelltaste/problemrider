---
title: Inadequate Integration Tests
description: The interactions between different modules or services are not thoroughly
  tested, leading to integration failures.
category:
- Architecture
- Code
- Testing
related_problems:
- slug: missing-end-to-end-tests
  similarity: 0.75
- slug: system-integration-blindness
  similarity: 0.75
- slug: poor-interfaces-between-applications
  similarity: 0.65
- slug: inadequate-test-infrastructure
  similarity: 0.6
- slug: quality-blind-spots
  similarity: 0.6
- slug: integration-difficulties
  similarity: 0.6
layout: problem
---

## Description

Inadequate integration tests occur when the testing strategy focuses primarily on individual components while failing to verify that different parts of the system work correctly together. Integration issues often arise at the boundaries between modules, services, or external systems, where assumptions about data formats, timing, error handling, or communication protocols may be incorrect. Without proper integration testing, systems may work well in isolation but fail when components interact in production environments.

## Indicators ⟡
- Unit tests pass but the application fails when modules are combined
- Bugs frequently occur at the boundaries between different system components
- Issues appear only when multiple features or services are used together
- Production problems often involve data format mismatches or communication failures
- Deployment to integrated environments reveals issues not caught in isolated testing

## Symptoms ▲

- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  Integration bugs that are not caught in testing escape to production, increasing the defect rate.
- [Regression Bugs](regression-bugs.md)
<br/>  Without integration tests, changes to one component can silently break interactions with other components.
- [Release Instability](release-instability.md)
<br/>  Releases that pass unit tests but lack integration coverage frequently cause production instability.
- [Cascade Failures](cascade-failures.md)
<br/>  Untested component interactions can trigger chain reactions of failures when assumptions at service boundaries are violated.
- [Frequent Hotfixes and Rollbacks](frequent-hotfixes-and-rollbacks.md)
<br/>  Integration issues discovered in production require emergency fixes and rollbacks to restore service.
## Causes ▼

- [Inadequate Test Infrastructure](inadequate-test-infrastructure.md)
<br/>  Lack of proper test environments and tools makes it difficult or impossible to run meaningful integration tests.
- [Inadequate Test Data Management](inadequate-test-data-management.md)
<br/>  Without realistic test data representing multi-component interactions, integration tests cannot effectively validate system behavior.
- [Time Pressure](time-pressure.md)
<br/>  Integration tests are more complex and time-consuming to write, so they are often skipped under deadline pressure.
- [Team Silos](team-silos.md)
<br/>  When teams work in isolation on individual components, nobody takes responsibility for testing cross-component interactions.
## Detection Methods ○
- **Integration Test Coverage Analysis:** Measure what percentage of component interactions are covered by integration tests
- **Production Issue Categorization:** Track how many bugs stem from integration problems versus component-specific issues
- **Interface Documentation Review:** Assess whether component interfaces are well-defined and tested
- **Cross-Component Bug Analysis:** Identify bugs that span multiple system components
- **Deployment Environment Testing:** Compare issue rates between isolated and integrated testing environments

## Examples

A microservices-based e-commerce platform has comprehensive unit tests for each service: user management, inventory, payment processing, and order fulfillment. Each service works perfectly in isolation and passes all unit tests. However, integration testing is minimal, focusing only on happy-path scenarios. In production, when a user attempts to purchase an out-of-stock item, the inventory service correctly returns an "out of stock" status, but the payment service has already processed the charge because it doesn't properly handle the timing of inventory checks. The order fulfillment service then fails because it receives conflicting information about payment status and inventory availability. The integration failure results in customers being charged for items they can't receive. Another example involves a document management system where the upload component, processing engine, and storage service all work correctly individually. However, integration testing missed the fact that the upload component produces metadata in one format while the processing engine expects a different format. In production, documents upload successfully and are stored correctly, but the processing engine silently fails to index them, making uploaded documents unsearchable despite appearing to be successfully processed.
