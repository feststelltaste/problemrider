---
title: Observability
description: Implementing structured logging, distributed tracing, and metrics for deep system understanding
category:
- Operations
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/observability/
problems:
- monitoring-gaps
- debugging-difficulties
- slow-incident-resolution
- constant-firefighting
- system-outages
- gradual-performance-degradation
- log-spam
- excessive-logging
- logging-configuration-issues
- insufficient-audit-logging
- log-injection-vulnerabilities
- slow-application-performance
- single-points-of-failure
- cascade-failures
- unpredictable-system-behavior
- increased-error-rates
layout: solution
---

## How to Apply ◆

> Adding observability to a legacy system means building the ability to ask new questions about system behavior — starting with the failure modes the team has already encountered but cannot currently explain.

- Begin with structured logging. Replace or supplement legacy free-text log output with machine-parseable structured log entries (JSON) that include at minimum: timestamp, severity, service/component name, correlation ID, and a brief event description. Many legacy systems log text to files with no structured fields — even adding a correlation ID field transforms the diagnostic capability.
- Introduce correlation IDs at the system boundary — the entry point where requests arrive. Pass this ID through all downstream calls, including calls to other legacy services, databases, and message queues. In legacy systems that span multiple codebases, this requires coordination but is the single most valuable observability improvement possible.
- Add the four golden signals as metrics for each major component: latency (with percentile breakdowns, not just averages), request/transaction rate, error rate, and resource saturation. For legacy batch systems, adapt these to batch-appropriate signals: records processed per run, failure rates, and processing lag.
- Use OpenTelemetry where possible to avoid vendor lock-in. Many legacy frameworks (Java Spring, .NET, older Python and Ruby frameworks) have OpenTelemetry agents or SDKs that provide automatic instrumentation with minimal code changes — HTTP calls, database queries, and common messaging libraries are often covered without writing instrumentation code.
- Prioritize instrumenting integration points first. Legacy systems typically fail at boundaries — when calling third-party APIs, when reading from shared databases, when consuming from queues. These are the hardest places to debug without traces and the places where visibility delivers the most immediate value.
- Establish Service Level Objectives (SLOs) early, even if informal. Without a definition of acceptable behavior, monitoring thresholds are arbitrary and alert fatigue is inevitable. SLOs focus the team's attention on the signals that actually represent user impact.
- Set up a centralized log aggregation and query platform (Elasticsearch/Kibana, Loki/Grafana, or a commercial equivalent). Legacy systems often have logs distributed across dozens of servers accessible only via SSH. Centralizing them changes incident investigation from a multi-hour multi-server exercise to a single query.
- Instrument business-level metrics alongside technical metrics. Legacy systems often have non-obvious business invariants (order counts per minute, transaction approval rates, batch completion times) that are more meaningful indicators of system health than CPU usage.

## Tradeoffs ⇄

> Observability transforms legacy system operations from reactive firefighting to data-driven diagnosis, but retrofitting instrumentation into existing systems is a sustained engineering investment rather than a one-time change.

**Benefits:**

- Incident investigation time drops significantly once correlation IDs and structured logs are in place. What previously required SSH access to multiple servers and manual log file scanning becomes a single query across a centralized log system.
- Teams gain the ability to diagnose novel failure modes without prior knowledge of that specific failure. Legacy systems regularly produce surprising behaviors; observability enables investigation of surprises rather than only recognition of previously seen patterns.
- Observability data reduces reliance on the tribal knowledge of senior engineers who have mentally modeled the system's behavior. Junior team members can investigate incidents independently using the same data the seniors use.
- Performance bottlenecks in legacy systems — often unknown because no measurement existed — become visible through distributed traces. Teams frequently discover that components they assumed were fast are responsible for significant latency.
- An observability layer built during modernization provides the verification signal for each change: teams can confirm empirically that a refactoring did not alter system behavior, rather than relying on test coverage that legacy systems often lack.

**Costs and Risks:**

- Retrofitting instrumentation into an established legacy codebase is labor-intensive, especially in systems with no existing logging conventions. Adding correlation ID propagation across a large, poorly structured codebase can require hundreds of small changes.
- Telemetry data volume for legacy systems under significant load can be enormous. Unsampled distributed traces, high-cardinality metrics, and verbose structured logs generate storage and processing costs that must be planned for. Unmanaged, these costs can be as large as the application infrastructure itself.
- Legacy systems often use older frameworks and libraries with limited or no OpenTelemetry support. Custom instrumentation must be written, adding development time and creating code that must be maintained alongside the legacy codebase.
- Alert fatigue is a particular risk when adding monitoring to a system that was previously unmonitored. Initial alert thresholds are often wrong, producing floods of false positives that teams learn to ignore — including alerts that eventually represent real problems.
- Teams need training on observability tools and investigation techniques. Adding Grafana dashboards and a Jaeger instance does not automatically improve incident response if the team does not know how to use them effectively under pressure.

## Examples

> Legacy systems are often the environments where observability delivers the most immediate operational value, precisely because they have historically been operated with the least visibility.

A manufacturing company's production scheduling system — a twelve-year-old Java monolith — was experiencing intermittent slowdowns that degraded factory floor scheduling for unpredictable periods of ten to forty minutes. The operations team had no instrumentation beyond basic CPU and memory graphs, and investigations always ended with "the system recovered on its own." After adding structured logging with correlation IDs and integrating OpenTelemetry's Java agent for automatic instrumentation, the team discovered within days that the slowdowns correlated with a specific combination of database query patterns that occurred when two particular scheduled jobs ran concurrently. The jobs had always run concurrently, but the slowdown only manifested when the production data volume exceeded a threshold that had been crossed in the previous year. Without the distributed traces, this connection would have remained invisible for years.

A large insurance company ran a COBOL-based claims processing system alongside a modern Java middleware layer that translated between the mainframe and web-facing services. Incidents at the integration boundary were common, and each one required a specialist team to manually correlate timestamps across mainframe SYSOUT logs and Java application logs — a process that took hours. The team introduced structured logging and correlation IDs in the Java middleware layer and built a log forwarding pipeline that included the COBOL system's job completion records. Suddenly, a single query across the aggregated logs could show the complete story of a claims processing failure from web request through Java middleware through mainframe job execution. Average time-to-diagnosis for integration incidents fell from four hours to under thirty minutes.

A financial trading platform running a mix of legacy C++ components and newer Python services had been struggling with unpredictable latency spikes at market open. Teams blamed different components in each incident. After instrumenting the system with Prometheus metrics and Grafana dashboards focused on the four golden signals for each component, a pattern emerged that had not been visible before: a legacy C++ order router was saturating its connection pool during the first two minutes of market open, causing back-pressure that propagated through the newer services in ways that looked like independent failures in each service's own metrics. The fix required changes to the legacy C++ component's connection pool configuration — a one-line change that nobody had found in three years of investigations because nobody had the data to point them there.
