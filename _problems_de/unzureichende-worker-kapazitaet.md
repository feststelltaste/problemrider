---
title: Unzureichende Worker-Kapazität
description: Es gibt nicht genug Worker-Prozesse oder -Threads, um das eingehende
  Aufgabenvolumen in einem asynchronen System zu handhaben, was zu wachsenden Warteschlangen
  führt.
category:
- Code
- Performance
related_problems:
- slug: growing-task-queues
  similarity: 0.75
- slug: task-queues-backing-up
  similarity: 0.75
- slug: work-queue-buildup
  similarity: 0.7
- slug: capacity-mismatch
  similarity: 0.65
- slug: thread-pool-exhaustion
  similarity: 0.6
- slug: staff-availability-issues
  similarity: 0.6
solutions:
- backpressure
- capacity-planning
- elastic-scaling
- parallelization
- monitoring-system-utilization
- load-testing
- performance-measurements
- load-shedding
- proactive-capacity-management
layout: problem
lang: de
en_slug: insufficient-worker-capacity
---

## Description
Unzureichende Worker-Kapazität ist ein verbreitetes Problem in Systemen, die ein Worker-Modell für asynchrone Verarbeitung nutzen. Wenn nicht genug Worker vorhanden sind, um das Volumen erzeugter Aufgaben zu handhaben, staut sich die Task-Queue auf, was zu Verzögerungen bei der Verarbeitung und potenziellem Datenverlust führt. Dies kann durch verschiedene Faktoren verursacht werden, von einem plötzlichen Traffic-Anstieg bis zu einer schrittweisen Zunahme der Arbeitslast über die Zeit. Die ordentliche Dimensionierung des Worker-Pools ist essenziell, um die Stabilität und Performance des Systems sicherzustellen.

## Indicators ⟡
- Die Anzahl der Nachrichten in der Warteschlange wächst.
- Die Zeit für die Verarbeitung einer Nachricht steigt.
- Die Worker laufen ständig mit hoher CPU- oder Speichernutzung.
- Es kommen Alerts vom Monitoring-System bezüglich der Queue-Größe.

## Symptoms ▲

- [Wachsende Task-Queues](wachsende-task-queues.md)
<br/>  Wenn Worker mit eingehenden Aufgaben nicht Schritt halten können, wachsen Warteschlangen kontinuierlich.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Aufgaben, die in aufgestauten Warteschlangen warten, verursachen merkliche Verzögerungen bei Anwendungsantworten und Verarbeitungszeiten.
- [Service-Timeouts](service-timeouts.md)
<br/>  Lange Wartezeiten in der Warteschlange lassen abhängige Dienste in ein Timeout laufen, während sie auf die Aufgabenfertigstellung warten.
- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Warteschlangenrückstau durch unzureichende Worker kann zu vorgelagerten Diensten kaskadieren, die von rechtzeitiger Verarbeitung abhängen.
- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Während die Arbeitslast wächst und die Worker-Kapazität konstant bleibt, verschlechtert sich die Systemperformance fortschreitend über die Zeit.

## Causes ▼

- [Kapazitäts-Fehlanpassung](kapazitaets-fehlanpassung.md)
<br/>  Eine fundamentale Fehlanpassung zwischen bereitgestellter Kapazität und tatsächlicher Nachfrage führt zu unzureichenden Workern.
- [Ineffizienzen bei der Skalierung](ineffizienzen-bei-der-skalierung.md)
<br/>  Die Unfähigkeit, Worker-Pools dynamisch als Reaktion auf Last zu skalieren, bedeutet, dass sich die Kapazität nicht an die Nachfrage anpassen kann.
- [Schlechte Planung](schlechte-planung.md)
<br/>  Das Versäumnis, für Arbeitslastwachstum zu planen, resultiert in Worker-Pools, die für die tatsächliche Nachfrage unterdimensioniert sind.

## Detection Methods ○

- **Queue-Monitoring:** Nachverfolgung von Queue-Größe, Nachrichtenraten und Consumer-Lag mit den Monitoring-Werkzeugen des Message-Queue-Systems.
- **Worker-Ressourcen-Monitoring:** Überwachung von CPU, Speicher und Netzwerk-I/O der Worker-Instanzen. Achten auf durchgängig hohe Auslastung.
- **Application Performance Monitoring (APM):** Nachverfolgung einzelner Task-Verarbeitungszeiten zur Identifikation, wo Verzögerungen innerhalb der Worker-Logik auftreten.
- **Lasttests:** Simulation von Spitzenlastbedingungen zur Identifikation des Punkts, an dem die Worker-Kapazität zum Engpass wird.
- **Log-Analyse:** Suche nach Logs, die auf Worker-Fehler, Wiederholungen oder ungewöhnlich lange Aufgabenzeiten hindeuten.

## Examples
Ein Bildverarbeitungsdienst nutzt eine Warteschlange für eingehende Bild-Upload-Anfragen. Anfänglich reicht eine Worker-Instanz aus. Während der Nutzer-Traffic wächst, beginnt sich die Warteschlange aufzustauen, und Bilder brauchen Stunden zur Verarbeitung. Das Hinzufügen weiterer Worker-Instanzen verringert sofort die Queue-Größe und Verarbeitungszeit. In einem anderen Fall ist ein Batch-Verarbeitungssystem so konfiguriert, dass es mit 4 Worker-Threads läuft. Eine neue, sehr CPU-intensive Berichtserstellungsaufgabe wird eingeführt. Wenn mehrere Berichtsanfragen gleichzeitig eintreffen, sind die 4 Threads vollständig ausgelastet, und nachfolgende Berichtsanfragen bleiben in der Warteschlange und warten, dass ein Thread frei wird. Dieses Problem ist grundlegend für skalierbare, asynchrone Systeme. Es unterstreicht die Notwendigkeit kontinuierlichen Monitorings und dynamischer Skalierungsstrategien, um die Verarbeitungskapazität mit der Nachfrage abzugleichen, besonders in Cloud-nativen Umgebungen.
