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
solutions:
- observability-and-monitoring
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
- [Negative User Feedback](negative-user-feedback.md)
<br/>  Users experiencing frequent errors provide negative feedback about system reliability and quality.
## Causes ▼

- [Inadequate Error Handling](inadequate-error-handling.md)
<br/>  Poor error handling allows failures to propagate rather than being caught and managed gracefully.
- [High Bug Introduction Rate](high-bug-introduction-rate.md)
<br/>  A high rate of new bugs being introduced with changes directly contributes to more runtime errors.
- [Incorrect Max Connection Pool Size](incorrect-max-connection-pool-size.md)
<br/>  Misconfigured connection pools cause connection exhaustion or rejection, generating application errors.
- [Database Connection Leaks](database-connection-leaks.md)
<br/>  Leaked connections exhaust the pool over time, causing increasing numbers of connection-related errors.
- [Service Timeouts](service-timeouts.md)
<br/>  Elevated error rates often accompany cascading failures that cause service timeouts across dependent systems.
- [Release Instability](release-instability.md)
<br/>  Spikes in error rates after deployments indicate releases are unstable and causing production problems.
## Detection Methods ○

- **Application Performance Monitoring (APM):** APM tools track error rates and can often pinpoint the exact line of code or service causing the error.
- **Log Aggregation and Analysis:** Centralized logging systems (e.g., ELK stack, Splunk) allow for easy searching, filtering, and visualization of error logs.
- **Metrics and Alerting:** Monitor error rates (e.g., HTTP 5xx errors, exception counts) and set up alerts for spikes.
- **Synthetic Monitoring:** Automated tests that simulate user interactions can detect errors before real users are affected.
- **User Feedback Channels:** Actively monitor customer support tickets, social media, and other feedback channels.

## Examples
After a new release, an e-commerce checkout service starts returning a high percentage of 500 errors. Investigation reveals a change in the payment gateway API, which the new code did not account for, leading to invalid requests. In another case, a microservice that processes image uploads suddenly sees a spike in errors. Upon investigation, it's found that the disk where uploaded images are stored has run out of space, causing file write operations to fail. Increased error rates are often the first symptom of a deeper underlying problem. Rapid detection and diagnosis are crucial to minimize impact on users and business operations.
