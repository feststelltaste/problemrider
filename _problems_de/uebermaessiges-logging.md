---
title: Übermäßiges Logging
description: Anwendungen erzeugen ein sehr hohes Volumen an Logs, was übermäßigen
  Speicherplatz verbraucht und potenziell die Performance beeinträchtigt.
category:
- Code
- Performance
related_problems:
- slug: log-spam
  similarity: 0.75
- slug: excessive-disk-io
  similarity: 0.75
- slug: slow-database-queries
  similarity: 0.7
- slug: high-database-resource-utilization
  similarity: 0.65
- slug: inefficient-code
  similarity: 0.65
- slug: memory-leaks
  similarity: 0.65
solutions:
- observability-and-monitoring
- asynchronous-logging
- platform-independent-logging-frameworks
- sampling
- error-logging
- logging-and-monitoring
- logging-guidelines
- monitoring
- code-reviews
- fast-feedback-loops
layout: problem
lang: de
en_slug: excessive-logging
---

## Description
Übermäßiges Logging kann erhebliche Auswirkungen auf Anwendungsperformance und Wartbarkeit haben. Wenn eine Anwendung zu viele Informationen protokolliert, kann dies eine große Menge an Speicherplatz verbrauchen, die Anwendung verlangsamen und es erschweren, wichtige Informationen in den Protokollen zu finden. Eine gut entworfene Logging-Strategie sollte sich darauf konzentrieren, nur die Informationen zu protokollieren, die für Debugging und Monitoring notwendig sind. Dies erfordert ein tiefes Verständnis der Anwendung und ihrer Anwendungsfälle.

## Indicators ⟡
- Protokolldateien wachsen mit unerwartet hoher Rate.
- Sie zahlen viel Geld für Protokollspeicherung und -analyse.
- Es ist schwierig, wichtige Informationen in Ihren Protokollen zu finden, aufgrund des Rauschens.
- Ihre Anwendung ist langsam, und Sie vermuten, dass Logging dazu beiträgt.

## Symptoms ▲

- [Übermäßige Festplatten-I/O](uebermaessige-festplatten-io.md)
<br/>  Hochvolumiges Logging erzeugt ständige Festplattenschreiboperationen, was erheblich zur gesamten Festplatten-I/O beiträgt.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Das Schreiben großer Protokollmengen, besonders synchron, verbraucht CPU- und I/O-Ressourcen, die die Hauptanwendung verlangsamen.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Das Schreiben von Protokollen konkurriert mit der Anwendungsverarbeitung um Festplatten-I/O-Bandbreite und CPU-Zyklen.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Wenn Protokolle zu viel Rauschen enthalten, wird das Finden relevanter Informationen für das Debugging zur Suche nach der Nadel im Heuhaufen.
- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Während sich Protokollvolumen im Laufe der Zeit anhäufen, füllt sich der Speicherplatz, und der I/O-Overhead wächst, was die Systemperformance progressiv verschlechtert.

## Causes ▼

- [Probleme bei der Logging-Konfiguration](probleme-bei-der-logging-konfiguration.md)
<br/>  Falsch konfigurierte Logging-Level, etwa das Belassen von DEBUG in der Produktion, verursachen direkt übermäßige Protokollausgabe.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Weniger erfahrene Entwickler neigen dazu, übermäßige Logging-Anweisungen als Debugging-Hilfe hinzuzufügen, ohne die Produktionsauswirkung zu berücksichtigen.
- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Schlechte Fehlerbehandlung, die vollständige Stack-Traces für jede Ausnahme protokolliert, trägt zur Explosion des Protokollvolumens bei.
- [Verworrene Querschnittsbelange](verworrene-querschnittsbelange.md)
<br/>  Wenn Logging in die Geschäftslogik verflochten ist, statt als Querschnittsbelang verwaltet zu werden, vermehren sich Logging-Anweisungen unkontrolliert.

## Detection Methods ○

- **Speichernutzungs-Monitoring:** Überwachung des Speicherplatzverbrauchs auf Servern, auf denen Protokolle gespeichert werden.
- **I/O-Monitoring:** Nutzung von Systemüberwachungswerkzeugen zur Nachverfolgung von Festplattenschreiboperationen im Zusammenhang mit Logging.
- **Protokollvolumenanalyse:** Nutzung von Protokollaggregationswerkzeugen zur Analyse des Protokollvolumens pro Anwendung oder Dienst.
- **Code-Review:** Suche nach Logging-Anweisungen, die übermäßig ausführlich sind oder in performancekritischen Abschnitten platziert wurden.
- **Konfigurations-Review:** Überprüfung von Logging-Konfigurationen, um sicherzustellen, dass angemessene Logging-Level für unterschiedliche Umgebungen gesetzt sind.

## Examples
Ein Microservice verarbeitet Millionen von Ereignissen pro Tag. Ein Entwickler setzt beim Debuggen eines Problems das Logging-Level auf `DEBUG` und vergisst, es vor dem Deployment in die Produktion zurückzusetzen. Innerhalb von Stunden ist der Speicherplatz des Servers vollständig durch Protokolldateien verbraucht, was den Dienst zum Absturz bringt. In einem anderen Fall protokolliert eine Anwendung die gesamte JSON-Payload jeder eingehenden API-Anfrage auf `INFO`-Ebene. Dies führt zu massiven Protokolldateien und erheblichem Netzwerkverkehr, wenn diese Protokolle an ein zentralisiertes Protokollierungssystem gesendet werden, obwohl nur ein kleiner Teil der Payload für die meisten Debugging-Zwecke relevant ist. Während Logging entscheidend für Observability ist, kann übermäßiges Logging zu einer Performance- und Kostenlast werden, was ein Gleichgewicht zwischen ausreichender Information und unnötigem Overhead erfordert.
