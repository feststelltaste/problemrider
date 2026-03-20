---
title: Error Reporting and Analysis
description: Systematic capture, analysis, and resolution of errors and issues
category:
- Process
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/error-reporting-and-analysis
problems:
- increased-error-rates
- slow-incident-resolution
- monitoring-gaps
- debugging-difficulties
- constant-firefighting
- high-defect-rate-in-production
- delayed-bug-fixes
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Implement an error tracking service (e.g., Sentry, Rollbar, Bugsnag) that captures, deduplicates, and groups errors automatically
- Instrument the legacy application to report unhandled exceptions and critical errors with full stack traces and context
- Define severity classifications and response time expectations for each severity level
- Create workflows that route error reports to the appropriate team based on the affected component
- Track error resolution metrics: time to acknowledge, time to resolve, and recurrence rate
- Conduct regular error trend reviews to identify systemic issues behind individual error reports
- Integrate error reporting with the issue tracking system so error patterns become actionable work items

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Transforms error handling from reactive incident response to systematic quality improvement
- Automatic deduplication and grouping prevents the same error from being investigated multiple times
- Provides data-driven prioritization of which errors have the greatest impact
- Creates accountability for error resolution through tracking and metrics
- Reduces time to resolution by providing complete error context upfront

**Costs and Risks:**
- Error tracking services add cost and require integration effort with legacy systems
- High error volumes can overwhelm teams if severity classification is not properly calibrated
- Over-reporting can cause alert fatigue, leading teams to ignore genuinely important errors
- Instrumenting legacy code for error reporting may require touching many files
- Privacy concerns if error reports capture user data

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy healthcare application had errors reported via a mix of email notifications, user support tickets, and developer observations. There was no unified view of error frequency or impact. The team integrated Sentry into the application, which immediately revealed that the top 10 error groups accounted for 80% of all production errors. Three of these were null reference errors in the patient scheduling module that had been reported as user complaints but never connected to code defects. By fixing just these three error groups over two sprints, the team reduced the overall production error rate by 60% and significantly decreased the support team's workload.
