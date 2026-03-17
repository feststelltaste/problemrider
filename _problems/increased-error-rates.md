---
title: Increased Error Rates
description: An unusual or sustained rise in the frequency of errors reported by an
  application or service.
category:
- Code
related_problems:
- slug: high-bug-introduction-rate
  similarity: 0.6
- slug: increased-bug-count
  similarity: 0.6
- slug: service-timeouts
  similarity: 0.55
- slug: upstream-timeouts
  similarity: 0.55
- slug: inadequate-error-handling
  similarity: 0.55
- slug: high-connection-count
  similarity: 0.55
layout: problem
---

## Description
An increased error rate is a clear sign that something is wrong with an application. This can be caused by a variety of factors, from a recent deployment that introduced a bug to a problem with a downstream service. A sudden spike in the error rate should be treated as a serious issue, as it can have a significant impact on the user experience and the stability of the system. A robust monitoring and alerting system is essential for detecting and responding to increased error rates in a timely manner.

## Indicators ⟡
- You are seeing a high number of errors in your logs.
- Your monitoring system is firing alerts for error thresholds being exceeded.
- You are getting complaints from users about errors.
- Your application is slow or unavailable.

## Symptoms ▲

- [Customer Dissatisfaction](customer-dissatisfaction.md)
<br/>  Frequent errors degrade the user experience, leading to frustration and loss of trust in the system.
- [Increased Customer Support Load](increased-customer-support-load.md)
<br/>  Users encountering errors contact support, driving up ticket volume.
- [Service Timeouts](service-timeouts.md)
<br/>  Elevated error rates often accompany cascading failures that cause service timeouts across dependent systems.
- [Negative User Feedback](negative-user-feedback.md)
<br/>  Users experiencing frequent errors provide negative feedback about system reliability and quality.
- [Release Instability](release-instability.md)
<br/>  Spikes in error rates after deployments indicate releases are unstable and causing production problems.

## Causes ▼
- [Inadequate Error Handling](inadequate-error-handling.md)
<br/>  Poor error handling allows failures to propagate rather than being caught and managed gracefully.
- [High Bug Introduction Rate](high-bug-introduction-rate.md)
<br/>  A high rate of new bugs being introduced with changes directly contributes to more runtime errors.
- [Incorrect Max Connection Pool Size](incorrect-max-connection-pool-size.md)
<br/>  Misconfigured connection pools cause connection exhaustion or rejection, generating application errors.
- [Database Connection Leaks](database-connection-leaks.md)
<br/>  Leaked connections exhaust the pool over time, causing increasing numbers of connection-related errors.
- [ABI Compatibility Issues](abi-compatibility-issues.md)
<br/>  Runtime failures from ABI mismatches lead to elevated error rates as function calls return unexpected values or crash.
- [Data Migration Integrity Issues](data-migration-integrity-issues.md)
<br/>  Migrated data with integrity issues triggers validation failures and application errors in the new system.
- [Dependency Version Conflicts](dependency-version-conflicts.md)
<br/>  Incompatible dependency versions cause unexpected runtime errors and method-not-found exceptions in production.
- [DMA Coherency Issues](dma-coherency-issues.md)
<br/>  Inconsistent memory views between CPU and DMA devices lead to sporadic errors in I/O operations, network processing, and data transfers.
- [Environment Variable Issues](environment-variable-issues.md)
<br/>  Missing or misconfigured environment variables lead to application failures and elevated error rates, especially after deployments.
- [Growing Task Queues](growing-task-queues.md)
<br/>  Tasks that age out or are retried excessively due to queue backlog generate elevated error rates.
- [High Connection Count](high-connection-count.md)
<br/>  Connection rejections when limits are reached cause application errors and failed requests.
- [Inconsistent Execution](inconsistent-execution.md)
<br/>  Manual, non-standardized execution leads to mistakes and omissions that produce more errors in the system.
- [Increased Manual Work](increased-manual-work.md)
<br/>  Manual execution of routine tasks is more error-prone than automated processes, leading to more mistakes.
- [Load Balancing Problems](load-balancing-problems.md)
<br/>  Overwhelmed instances from uneven load distribution start dropping requests or returning errors.
- [Manual Deployment Processes](manual-deployment-processes.md)
<br/>  Human execution of deployment steps inevitably introduces errors that automated processes would avoid.
- [Null Pointer Dereferences](null-pointer-dereferences.md)
<br/>  Null pointer exceptions manifest as runtime errors that increase the overall error rate of the application.
- [Poor Interfaces Between Applications](poor-interfaces-between-applications.md)
<br/>  Poorly defined interfaces produce frequent integration errors from mismatched data formats and inconsistent contracts.
- [Service Discovery Failures](service-discovery-failures.md)
<br/>  Services unable to discover their dependencies generate connection errors and service-not-found errors at elevated rates.
- [Service Timeouts](service-timeouts.md)
<br/>  Timeout errors contribute directly to elevated error rates across the system as requests fail to complete.
- [Upstream Timeouts](upstream-timeouts.md)
<br/>  Timeout errors directly increase the overall error rate of the system as requests fail without receiving responses.

## Detection Methods ○

- **Application Performance Monitoring (APM):** APM tools track error rates and can often pinpoint the exact line of code or service causing the error.
- **Log Aggregation and Analysis:** Centralized logging systems (e.g., ELK stack, Splunk) allow for easy searching, filtering, and visualization of error logs.
- **Metrics and Alerting:** Monitor error rates (e.g., HTTP 5xx errors, exception counts) and set up alerts for spikes.
- **Synthetic Monitoring:** Automated tests that simulate user interactions can detect errors before real users are affected.
- **User Feedback Channels:** Actively monitor customer support tickets, social media, and other feedback channels.

## Examples
After a new release, an e-commerce checkout service starts returning a high percentage of 500 errors. Investigation reveals a change in the payment gateway API, which the new code did not account for, leading to invalid requests. In another case, a microservice that processes image uploads suddenly sees a spike in errors. Upon investigation, it's found that the disk where uploaded images are stored has run out of space, causing file write operations to fail. Increased error rates are often the first symptom of a deeper underlying problem. Rapid detection and diagnosis are crucial to minimize impact on users and business operations.
