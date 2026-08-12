---
title: Logging Guidelines
description: Agree what gets logged at which level, what must never be logged, and how long it is kept — so that logs become a diagnostic tool rather than a volume problem.
category:
- Code
- Operations
problems:
- excessive-logging
- log-spam
- logging-configuration-issues
- log-injection-vulnerabilities
- monitoring-gaps
- inadequate-error-handling
- debugging-difficulties
- operational-overhead
- excessive-disk-io
- resource-waste
layout: solution
---

## Description

Logging guidelines are a short agreement about what the system writes to its logs: which events warrant which level, what context every entry must carry, what must never appear, and how long entries are retained. They address a failure mode that is almost universal in long-lived systems and rarely treated as a problem in its own right. Logging is added incrementally by many people over many years with no shared convention, and it converges on the worst of both outcomes: enormous volume and no diagnostic value. Everything is logged at the same level, entries lack the identifiers needed to correlate them, the same event appears three times with different wording, and the one thing needed to diagnose the current incident was never logged at all. Guidelines cost almost nothing and convert logs from a storage line item into the primary instrument for understanding a system nobody fully understands.

## How to Apply ◆

> In a legacy system logs are frequently the only observability that exists, which makes their quality the limiting factor on every diagnosis.

- **Define what each level means** in terms of who acts on it: error means someone must intervene and something is broken, warning means something unexpected happened that the system handled, info records significant state transitions, and debug is for development only. Without stated definitions everything becomes an error or everything becomes info, and both are equivalent to no levels at all.
- **Require a correlation identifier** on every entry, propagated across service and thread boundaries. Without it, logs are a chronological mixture of unrelated work, and reconstructing one request's path through the system is the single most common diagnostic need.
- **Log structured data, not sentences.** Key-value or JSON entries can be filtered and aggregated; prose entries can only be read by eye, which does not scale past one incident. This single change usually does more for diagnostic value than any amount of additional logging.
- **State what must never be logged**: credentials, tokens, personal data beyond what is necessary, payment details, and full request bodies for anything sensitive. Log content is copied to aggregation systems, backups, and support tickets, and it is a routine source of data exposure.
- **Neutralize untrusted input before logging it.** Values that reach the log from user input can inject newlines and forge entries, or exploit the log viewer. Escaping logged values is cheap and prevents a class of vulnerability that is easy to overlook.
- **Log the context needed to act, not the fact of an occurrence.** "Payment validation failed" is not actionable; the same entry with the rule that rejected it, the input class, and the correlation identifier is. Most legacy logging is voluminous and contextless.
- **Set retention by level and by value**, with the cost stated. Debug entries retained for a year are a storage bill with no purpose; error entries deleted after a week make trend analysis impossible.
- **Make levels configurable at runtime** without a deployment. The ability to raise verbosity for one component during an incident and lower it afterward is what allows a low default volume, which is what keeps logs usable.
- **Prune during review.** Ask whether each new log entry would be read by anyone, and delete entries that have never been useful. Logging is the only code that is never removed, because removing it feels risky and adding it feels responsible.
- **Watch the performance cost.** Synchronous logging in a hot path is a genuine bottleneck, and excessive logging can consume more resources than the work being logged.

## Tradeoffs ⇄

> Guidelines make logs usable and cheaper, but retrofitting them across an old codebase is significant work and reducing volume always risks removing something that would have mattered.

**Benefits:**

- Diagnosis becomes substantially faster, since correlated structured entries allow a request's path to be reconstructed instead of inferred.
- Volume and storage cost fall, often dramatically, because most of the volume in an unmanaged system carries no information.
- Alerting becomes possible. An error level that genuinely means something broken can be alerted on; one that fires thousands of times a day cannot.
- Sensitive data exposure through logs is prevented, closing a channel that is easy to overlook precisely because logs are not thought of as data stores.
- Log injection is closed off, a vulnerability class that is cheap to prevent and awkward to detect after the fact.

**Costs and Risks:**

- Retrofitting guidelines across a large legacy codebase is a large mechanical task with no visible output, and it is rarely completed.
- Reducing volume can remove an entry that would have been decisive in some future incident, and that loss is only ever discovered later.
- Structured logging requires framework support and sometimes a migration, which is real effort in older stacks.
- Runtime-configurable levels add configuration surface, and a misconfiguration can silence logging entirely — a failure that is invisible until something goes wrong.
- Guidelines applied inconsistently produce a codebase with two logging conventions, which for filtering and aggregation purposes can be worse than one bad convention applied uniformly.

## How It Could Be

A team maintaining a payments platform generated roughly 400 gigabytes of logs a day, at meaningful cost, and still routinely could not diagnose failures. Investigation found that one entry — written at error level on every retryable timeout, which occurred normally — accounted for about 60 percent of the volume. Nothing carried a correlation identifier, so reconstructing a single failed payment meant grepping by timestamp across four services and guessing. They wrote a two-page guideline: four defined levels, a mandatory correlation identifier, structured entries, and a prohibited-content list. Retrofitting the correlation identifier took three weeks. Volume fell to about 40 gigabytes a day, and the median time to diagnose a payment failure fell from over two hours to roughly fifteen minutes — almost entirely because a single filter now returned one payment's complete path.

The prohibited-content list found something the team had not been looking for. A search of the log aggregation system for the newly forbidden patterns turned up a debug entry, added four years earlier during an integration problem and never removed, that logged complete inbound API requests including authentication headers. Those logs were retained for 90 days and were readable by everyone with access to the aggregation tool, which was most of the engineering organization. The entry was removed and the retained logs purged, and the incident became the reason the organization added a recurring automated scan of log content for credential patterns.
