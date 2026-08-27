---
title: Sampling
description: Nutzung einer repräsentativen Teilmenge von Daten für Analyse
  oder Tests.
category:
- Performance
- Testing
problems:
- unbounded-data-growth
- slow-database-queries
- high-database-resource-utilization
- slow-application-performance
- inadequate-test-data-management
- excessive-logging
layout: solution
lang: de
en_slug: sampling
related_solutions:
- slug: distributed-processing
  similarity: 0.75
- slug: data-archiving
  similarity: 0.7
- slug: logging
  similarity: 0.7
- slug: data-replication
  similarity: 0.7
- slug: compression
  similarity: 0.7
- slug: data-partitioning
  similarity: 0.7
---

## Description

Sampling verarbeitet und analysiert eine repräsentative Teilmenge von Daten statt des vollständigen Datensatzes, unter Nutzung einer Strategie — zufällig, geschichtet oder Reservoir-Sampling unter anderen —, gewählt passend zu den statistischen Anforderungen der Aufgabe, und angewendet am Punkt der Datensammlung, sodass von Anfang an nur die notwendige Teilmenge erfasst wird. Dies ist besonders effektiv für Arbeitslasten wie Monitoring, Trendanalyse und Testing, wo die Verarbeitung jedes einzelnen Datenpunkts Kosten hinzufügt, ohne proportionale Erkenntnis hinzuzufügen, und wo eine gut gewählte Stichprobe statistisch nicht von der Analyse des vollständigen Datensatzes unterscheidbare Schlussfolgerungen liefert. Legacy-Systeme sammeln häufig Monitoring-, Logging- und Tracing-Daten in einem Volumen an, das nie antizipiert wurde, als der ursprüngliche Sammelmechanismus entworfen wurde, und bis dies zu einem Problem wird, ist die erschöpfende Sammelgewohnheit oft zu tief in das operative Tooling des Systems eingebettet, um sie einfach abzuschalten; Sampling bietet einen Weg, dieses Volumen dramatisch zu reduzieren, ohne Observability gänzlich aufzugeben. Es ist besonders nützlich kombiniert mit Schichtung, die vollständige Erfassung der seltensten und wichtigsten Ereignisse garantiert — wie das Erfassen von 100 Prozent der Fehler-Traces, während nur ein kleiner Bruchteil erfolgreicher gesampelt wird —, sodass die exakten Fälle, die für Debugging am wertvollsten sind, nie diejenigen sind, die der Reduktion zum Opfer fallen. Da gesampelte Ergebnisse Näherungswerte statt exakter Zahlen sind, müssen die Methodik und ihre Konfidenzintervalle dokumentiert und periodisch gegen Vollständige-Daten-Analyse validiert werden, sodass Konsumenten der gesampelten Daten ihre Beschränkungen verstehen, statt sie für eine vollständige Aufzeichnung zu halten.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Arbeitslasten, bei denen die Verarbeitung von 100 Prozent der Daten unnötig ist: Analytics, Monitoring, Trenderkennung, Testing
- Wählen Sie eine angemessene Sampling-Strategie (zufällig, geschichtet, Reservoir) basierend auf den statistischen Anforderungen
- Implementieren Sie Sampling am Punkt der Datensammlung statt alles zu sammeln und später zu filtern
- Nutzen Sie geschichtetes Sampling, wenn verschiedene Datensegmente unterschiedliche Wichtigkeit oder Varianz haben
- Wenden Sie Sampling auf verteiltes Tracing und Logging an, um Speicherkosten zu reduzieren, während diagnostische Fähigkeit erhalten bleibt
- Validieren Sie, dass gesampelte Ergebnisse statistisch repräsentativ bleiben, durch periodischen Vergleich mit Vollständige-Daten-Analyse
- Dokumentieren Sie die Sampling-Methodik und Konfidenzintervalle, sodass Konsumenten die Beschränkungen der Daten verstehen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert Verarbeitungszeit, Speicherkosten und Infrastrukturanforderungen dramatisch
- Macht Echtzeitanalyse für Datensätze machbar, die zu groß für erschöpfende Verarbeitung sind
- Reduziert Log-Speicherkosten bei Erhalt ausreichender Daten für Fehlersuche
- Ermöglicht schnellere Testzyklen durch Arbeit mit handhabbaren Datenteilmengen

**Kosten und Risiken:**
- Seltene Ereignisse könnten übersehen werden, wenn die Stichprobengröße zu klein oder das Sampling nicht geschichtet ist
- Ergebnisse sind näherungsweise und erfüllen möglicherweise nicht Audit- oder Compliance-Anforderungen
- Falsche Sampling-Methodik kann systematischen Bias einführen
- Teams könnten die Beschränkungen gesampelter Daten nicht verstehen und sie als exakt behandeln
- Das Debuggen spezifischer Produktionsprobleme ist schwerer, wenn der relevante Trace nicht gesampelt wurde

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Monitoring-System sammelte und speicherte jeden einzelnen Anfrage-Trace, verbrauchte täglich 2 TB Speicher und machte Trace-Suche unerschwinglich langsam. Das Team implementierte adaptives Sampling, das 100 Prozent der Fehler-Traces und 1 Prozent der erfolgreichen Traces erfasste, mit geschichtetem Sampling, das sicherstellte, dass jeder Endpunkt unabhängig vom Traffic-Volumen repräsentiert war. Dies reduzierte den Speicher auf 50 GB pro Tag und machte Trace-Suche reaktionsschnell, während die 100%ige Fehlererfassung sicherstellte, dass keine für Debugging kritischen Daten verloren gingen. Monatliche statistische Vergleiche bestätigten, dass die gesampelten Latenzverteilungen innerhalb von 2 Prozent der wahren Werte blieben.
