---
title: Wachsende Task-Queues
description: Asynchrone Verarbeitungswarteschlangen häufen unverarbeitete Aufgaben
  an, was auf einen Engpass in der Verarbeitungspipeline hindeutet.
category:
- Code
- Performance
related_problems:
- slug: task-queues-backing-up
  similarity: 0.8
- slug: insufficient-worker-capacity
  similarity: 0.75
- slug: work-queue-buildup
  similarity: 0.7
- slug: service-timeouts
  similarity: 0.6
- slug: increased-error-rates
  similarity: 0.6
- slug: external-service-delays
  similarity: 0.6
solutions:
- backpressure
- capacity-planning
- elastic-scaling
- asynchronous-processing
- batch-processing
- data-stream-processing
- parallelization
- pipelining
- streaming
layout: problem
lang: de
en_slug: growing-task-queues
---

## Description
Eine wachsende Task-Queue ist ein klares Zeichen dafür, dass ein System mit seiner Arbeitslast nicht Schritt halten kann. Wenn Aufgaben schneller erzeugt als konsumiert werden, wächst die Warteschlange, was zu Verzögerungen bei der Verarbeitung und potenziellem Datenverlust führt. Dies kann durch verschiedene Faktoren verursacht werden, von einem plötzlichen Traffic-Anstieg bis zu einer schrittweisen Zunahme der Arbeitslast über die Zeit. Ein robustes Monitoring- und Alerting-System ist essenziell, um eine wachsende Task-Queue zeitnah zu erkennen und darauf zu reagieren.

## Indicators ⟡
- Die Zeit für die Verarbeitung einer Aufgabe steigt schrittweise.
- Die Anzahl der Worker-Prozesse reicht nicht aus, um die Last zu bewältigen.
- Es zeigt sich eine Zunahme der Anzahl wiederholter Aufgaben.
- Es kommen Alerts vom Monitoring-System bezüglich der Queue-Größe.

## Symptoms ▲

- [Service-Timeouts](service-timeouts.md)
<br/>  Aufgaben, die zu lange in Warteschlangen warten, überschreiten Timeout-Schwellenwerte, bevor sie verarbeitet werden können.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Nutzer erleben Verzögerungen, während ihre Anfragen in wachsenden Warteschlangen auf Verarbeitung warten.
- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Aufgaben, die veralten oder aufgrund von Warteschlangenrückstau übermäßig wiederholt werden, erzeugen erhöhte Fehlerraten.
- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Warteschlangenrückstau kann Systemressourcen erschöpfen und kaskadierende Ausfälle über abhängige Dienste hinweg erzeugen.
- [Negatives Nutzerfeedback](negatives-nutzerfeedback.md)
<br/>  Nutzer beklagen sich über verzögerte Verarbeitung von Operationen wie E-Mail-Bestätigungen und Bestellverarbeitung.
- [Nutzerfrustration](nutzerfrustration.md)
<br/>  Nutzer erleben Verzögerungen, während ihre Anfragen in wachsenden Warteschlangen liegen.

## Causes ▼

- [Unzureichende Worker-Kapazität](unzureichende-worker-kapazitaet.md)
<br/>  Nicht genügend Worker-Prozesse, um Aufgaben in der Rate zu konsumieren, in der sie erzeugt werden, verursacht direkt Queue-Wachstum.
- [Ineffizienter Code](ineffizienter-code.md)
<br/>  Langsamer Task-Verarbeitungscode bedeutet, dass jeder Worker länger pro Aufgabe braucht, was den Gesamtdurchsatz beim Konsumieren verringert.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Worker, die um begrenzte CPU-, Speicher- oder I/O-Ressourcen konkurrieren, verarbeiten Aufgaben langsamer, wodurch Warteschlangen wachsen können.
- [Verzögerungen durch externe Services](verzoegerungen-durch-externe-services.md)
<br/>  Worker, die blockiert sind, während sie auf langsame externe Dienste warten, verringern den Verarbeitungsdurchsatz und verursachen Warteschlangenansammlung.

## Detection Methods ○

- **Queue-Monitoring:** Nutzung der von Message-Queue-Systemen bereitgestellten Monitoring-Werkzeuge (z. B. RabbitMQ Management, Kafka Metrics, AWS SQS/SNS-Metriken), um Queue-Größe, Nachrichtenraten und Consumer-Lag nachzuverfolgen.
- **Worker-Prozess-Monitoring:** Überwachung von CPU-, Speicher- und I/O-Nutzung der Worker-Prozesse.
- **Distributed Tracing:** Nachverfolgung asynchroner Operationen zur Identifikation von Engpässen innerhalb der Worker-Verarbeitungslogik oder externer Abhängigkeiten.
- **Log-Analyse:** Suche nach Fehlern oder Warnungen in Worker-Logs, die auf Verarbeitungsfehler oder Wiederholungen hindeuten.

## Examples
Eine E-Commerce-Website nutzt eine Message-Queue, um Bestellbestätigungen zu verarbeiten und E-Mails zu senden. Während eines Flash-Sales steigt die Anzahl der Bestellungen sprunghaft an, und die E-Mail-Warteschlange wächst schnell. Kunden beklagen sich, dass sie ihre Bestellbestätigungen stundenlang nicht erhalten, weil die E-Mail-Sende-Worker nicht mithalten können. In einem anderen Fall nutzt eine Datenverarbeitungspipeline eine Warteschlange zur Handhabung von Bildgrößenänderungsaufgaben. Ein neues, sehr großes Bildformat wird eingeführt, und die zuvor effizienten Bildgrößenänderungs-Worker brauchen jetzt viel länger pro Bild, was zu einem Rückstau der Warteschlange führt. Dieses Problem ist in ereignisgesteuerten Architekturen und Microservices verbreitet, in denen asynchrone Verarbeitung stark genutzt wird. Es unterstreicht die Wichtigkeit ordentlicher Kapazitätsplanung und robuster Fehlerbehandlung für Hintergrundaufgaben.
