---
title: Growing Task Queues
description: Asynchronous processing queues accumulate unprocessed tasks, indicating
  a bottleneck in the processing pipeline.
category:
- Code
- Performance
related_problems:
- slug: insufficient-worker-capacity
  similarity: 0.85
- slug: task-queues-backing-up
  similarity: 0.85
- slug: work-queue-buildup
  similarity: 0.7
- slug: unbounded-data-growth
  similarity: 0.5
- slug: increased-error-rates
  similarity: 0.5
- slug: thread-pool-exhaustion
  similarity: 0.5
layout: problem
---

## Description
A growing task queue is a clear sign that a system is not able to keep up with its workload. When tasks are produced faster than they are consumed, the queue will grow, leading to delays in processing and a potential for data loss. This can be caused by a variety of factors, from a sudden spike in traffic to a gradual increase in the workload over time. A robust monitoring and alerting system is essential for detecting and responding to a growing task queue in a timely manner.

## Indicators ⟡
- The time it takes to process a task is gradually increasing.
- The number of worker processes is not sufficient to handle the load.
- You are seeing an increase in the number of tasks that are being retried.
- You are getting alerts from your monitoring system about the queue size.

## Symptoms ▲

- [Service Timeouts](service-timeouts.md)
<br/>  Tasks waiting too long in queues exceed timeout thresholds before they can be processed.
- [Slow Application Performance](slow-application-performance.md)
<br/>  Users experience delays as their requests wait in growing queues before being processed.
- [Increased Error Rates](increased-error-rates.md)
<br/>  Tasks that age out or are retried excessively due to queue backlog generate elevated error rates.
- [Cascade Failures](cascade-failures.md)
<br/>  Queue buildup can exhaust system resources and create cascading failures across dependent services.
- [Negative User Feedback](negative-user-feedback.md)
<br/>  Users complain about delayed processing of operations like email confirmations and order processing.

## Causes ▼
- [Insufficient Worker Capacity](insufficient-worker-capacity.md)
<br/>  Not enough worker processes to consume tasks at the rate they are produced directly causes queue growth.
- [Inefficient Code](inefficient-code.md)
<br/>  Slow task processing code means each worker takes longer per task, reducing overall consumption throughput.
- [Resource Contention](resource-contention.md)
<br/>  Workers competing for limited CPU, memory, or I/O resources process tasks more slowly, allowing queues to grow.
- [External Service Delays](external-service-delays.md)
<br/>  Workers blocked waiting for slow external services reduce processing throughput and cause queue accumulation.

## Detection Methods ○

- **Queue Monitoring:** Use the monitoring tools provided by the message queue system (e.g., RabbitMQ Management, Kafka Metrics, AWS SQS/SNS metrics) to track queue size, message rates, and consumer lag.
- **Worker Process Monitoring:** Monitor the CPU, memory, and I/O usage of worker processes.
- **Distributed Tracing:** Trace asynchronous operations to identify bottlenecks within the worker processing logic or external dependencies.
- **Log Analysis:** Look for errors or warnings in worker logs that indicate processing failures or retries.

## Examples
An e-commerce site uses a message queue to process order confirmations and send emails. During a flash sale, the number of orders spikes, and the email queue grows rapidly. Customers complain about not receiving their order confirmations for hours, because the email sending workers cannot keep up. In another case, a data processing pipeline uses a queue to handle image resizing tasks. A new, very large image format is introduced, and the image resizing workers, which were previously efficient, now take much longer per image, causing the queue to back up. This problem is common in event-driven architectures and microservices where asynchronous processing is heavily utilized. It highlights the importance of proper capacity planning and robust error handling for background tasks.
