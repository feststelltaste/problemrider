---
title: Smoke Testing
description: Performing a series of basic tests to verify the core functionality of a system
category:
- Testing
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/smoke-testing
problems:
- frequent-hotfixes-and-rollbacks
- regression-bugs
- deployment-risk
- high-defect-rate-in-production
- insufficient-testing
- release-instability
- fear-of-change
- missing-end-to-end-tests
- legacy-code-without-tests
layout: solution
---

## How to Apply ◆

> Legacy systems often lack comprehensive test suites, making every deployment a gamble. Smoke tests provide a lightweight safety net that verifies core functionality works after deployment, catching catastrophic failures before they reach users.

- Identify the 10-20 most critical user-facing operations in the legacy system — login, core transactions, key API endpoints, critical batch jobs. These form the initial smoke test suite and should cover the paths that, if broken, would make the system unusable.
- Implement smoke tests as simple, fast, end-to-end checks that verify each critical operation completes successfully. Smoke tests should not be exhaustive; they should run in under 5 minutes and confirm that the system is fundamentally operational.
- Integrate smoke tests into the deployment pipeline so they run automatically after every deployment to each environment. A failed smoke test should block promotion to the next environment or trigger an automatic rollback.
- Design smoke tests to be environment-agnostic by parameterizing endpoints, credentials, and test data. This allows the same suite to run against staging, pre-production, and production environments.
- Include health checks for critical dependencies (database connectivity, external service availability, message broker connectivity) as part of the smoke suite. Legacy systems often fail silently when a dependency is unavailable.
- Add smoke tests for legacy batch processes by verifying that a minimal batch run completes without errors. Batch failures in legacy systems often go undetected until downstream consumers notice missing data hours later.
- Schedule periodic smoke test execution in production (every 15-30 minutes) to detect environmental issues, certificate expirations, or resource exhaustion before users are affected.

## Tradeoffs ⇄

> Smoke tests provide rapid, high-confidence verification of core functionality with minimal investment, but they are deliberately shallow and cannot replace comprehensive testing.

**Benefits:**

- Catch catastrophic deployment failures within minutes rather than hours, drastically reducing mean time to detection for broken deployments.
- Provide a low-cost starting point for testing legacy systems that have no existing test suite, delivering immediate value with minimal effort.
- Build deployment confidence incrementally, helping teams move from fear-driven manual verification to automated safety checks.
- Enable faster rollback decisions by providing clear pass/fail signals immediately after deployment.

**Costs and Risks:**

- Smoke tests only verify that the system starts and handles basic operations; they miss subtle bugs, performance regressions, and edge cases that require deeper testing.
- False confidence can develop if teams treat passing smoke tests as proof that a deployment is safe, neglecting the need for broader test coverage.
- Smoke tests in production require careful design to avoid side effects — creating test data, triggering notifications, or affecting real users.
- Maintaining smoke tests for rapidly evolving systems requires ongoing effort to keep them aligned with current functionality.

## Examples

> The following scenarios illustrate how smoke tests catch critical failures in legacy systems.

A government agency deploys updates to a legacy citizen services portal monthly. Each deployment involves updating configuration files, database schemas, and application binaries across multiple servers. On three occasions in the past year, deployments broke the login flow due to misconfigured authentication settings, but the issue was not discovered until citizens called the help desk the next morning. The team implements a smoke test suite that, immediately after deployment, attempts to log in with a test account, submit a sample application form, and verify that the main dashboard loads correctly. During the next deployment, the smoke test detects that the login endpoint returns a 500 error due to a missing environment variable. The deployment is rolled back within 10 minutes, before any citizen is affected. The entire smoke suite runs in 90 seconds.

A manufacturing company runs a legacy ERP system where nightly batch jobs process production orders, inventory updates, and supplier invoices. Failures in these batch jobs are typically discovered the next afternoon when warehouse staff notice missing pick lists. The team creates a smoke test that runs after each batch cycle, verifying that at least one order was processed, inventory counts were updated, and no error records were written to the batch log table. When a database connection pool misconfiguration causes the inventory batch to silently fail, the smoke test detects the zero-update condition within 15 minutes and pages the on-call engineer, who fixes the issue before the morning shift begins.
