---
title: Error Reporting and Analysis
description: Systematic capture, analysis, and resolution of errors and issues
category:
- Process
- Operations
problems:
- increased-error-rates
- slow-incident-resolution
- monitoring-gaps
- debugging-difficulties
- constant-firefighting
- high-defect-rate-in-production
- delayed-bug-fixes
- delayed-issue-resolution
layout: solution
related_solutions:
- slug: error-logs
  similarity: 0.9
- slug: error-handling
  similarity: 0.85
- slug: error-logging
  similarity: 0.85
- slug: root-cause-analysis
  similarity: 0.8
- slug: exceptions
  similarity: 0.8
- slug: logging
  similarity: 0.8
---

## Description

Error reporting and analysis introduces dedicated tooling — services like Sentry, Rollbar, or Bugsnag — that automatically captures unhandled exceptions and critical errors with full context, deduplicates and groups occurrences of the same underlying defect, and routes them through a defined workflow with severity classifications, ownership, and resolution tracking. This goes beyond raw logging by turning individual error occurrences into managed issues: instead of a stream of log lines that has to be manually correlated, the team sees a ranked list of distinct error groups with frequency and impact data attached. Legacy systems that previously relied on an informal mix of user complaints, support tickets, and developer observations to learn about production problems typically discover, once such tooling is introduced, that a small number of error groups account for the overwhelming majority of production failures — defects that had been reported piecemeal as user complaints for a long time without anyone connecting them to a single root cause. Because this requires instrumenting the legacy application to emit rich error reports and integrating a paid or self-hosted tracking service, the main costs are the integration effort itself and the need to calibrate severity classification carefully enough to avoid either alert fatigue or missed critical issues, along with attention to what user data ends up captured in the reports.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy healthcare application had errors reported via a mix of email notifications, user support tickets, and developer observations. There was no unified view of error frequency or impact. The team integrated Sentry into the application, which immediately revealed that the top 10 error groups accounted for 80% of all production errors. Three of these were null reference errors in the patient scheduling module that had been reported as user complaints but never connected to code defects. By fixing just these three error groups over two sprints, the team reduced the overall production error rate by 60% and significantly decreased the support team's workload.
