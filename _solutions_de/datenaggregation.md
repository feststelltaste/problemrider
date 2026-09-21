---
title: Datenaggregation
description: Zusammenfassung feingranularer Daten zu kompakteren Einheiten.
category:
- Performance
- Database
problems:
- slow-database-queries
- unbounded-data-growth
- high-database-resource-utilization
- gradual-performance-degradation
- database-query-performance-issues
layout: solution
lang: de
en_slug: data-aggregation
related_solutions:
- slug: materialized-views
  similarity: 0.75
- slug: data-partitioning
  similarity: 0.7
- slug: data-archiving
  similarity: 0.7
- slug: denormalization
  similarity: 0.7
- slug: data-replication
  similarity: 0.65
- slug: sampling
  similarity: 0.65
---

## Description

Datenaggregation fasst feingranulare Datensätze zu kompakten, vorberechneten Einheiten zusammen — stündliche, tägliche oder monatliche Summen, Durchschnitte oder Zählungen —, sodass Abfragen, die einen Überblick brauchen, nicht mehr jedes Mal das volle Volumen an Rohdaten scannen und neu berechnen müssen. Der Mechanismus trennt das Anliegen der Sammlung detaillierter Daten vom Anliegen der Beantwortung zusammenfassender Fragen darüber: Detaillierte Zeilen häufen sich weiterhin an der Quelle an, während ein separater Aggregationsprozess sie periodisch oder inkrementell in Sekundärtabellen oder materialisierte Views zusammenfasst, die auf bekannte Berichtsbedürfnisse zugeschnitten sind. In Legacy-Systemen ist dies wichtig, weil Jahre unkontrollierten Datenwachstums kombiniert mit Berichtsabfragen, die für einen viel kleineren Datensatz entworfen wurden, routinemäßig Dashboards und periodische Berichte in mehrminütige Operationen verwandeln, die mit transaktionalen Workloads um dieselben Datenbankressourcen konkurrieren. Aggregation lässt ein Legacy-System seine detaillierte Historie für Audit- und Drilldown-Zwecke intakt behalten, während Berichtskonsumenten einen Datensatz erhalten, der um Größenordnungen kleiner und günstiger abzufragen ist. Weil die Aggregate abgeleitet statt maßgeblich sind, bieten sie auch eine natürliche Grenze, an der feingranulare Daten später archiviert oder verworfen werden können, ohne die Zusammenfassungen zu beeinträchtigen, auf die sich die meisten Konsumenten tatsächlich verlassen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie Abfragen, die große Mengen feingranularer Daten scannen, um zusammenfassende Ergebnisse zu erzeugen
- Erstellen Sie vorab aggregierte Tabellen oder materialisierte Views für gängige Berichtszeiträume (stündlich, täglich, monatlich)
- Implementieren Sie inkrementelle Aggregation, die nur neue Daten verarbeitet, statt von Grund auf neu zu berechnen
- Planen Sie Aggregationsjobs während schwachlastiger Zeiten, um die Auswirkung auf transaktionale Workloads zu minimieren
- Definieren Sie Aufbewahrungsrichtlinien: Behalten Sie feingranulare Daten für einen begrenzten Zeitraum und aggregierte Daten länger
- Nutzen Sie die aggregierten Daten für Dashboards und Berichte, während detaillierte Daten für Drilldown bei Bedarf erhalten bleiben

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verbessert die Abfrageperformance für Zusammenfassungs- und Berichts-Anwendungsfälle dramatisch
- Reduziert das Datenvolumen, das für Analyseabfragen gescannt werden muss
- Ermöglicht schnelles Dashboard-Rendering selbst über große historische Datensätze
- Reduziert Speicherwachstum, wenn kombiniert mit der Archivierung feingranularer Daten

**Kosten und Risiken:**
- Aggregierte Daten verlieren Detail, was Ad-hoc-Untersuchungen einzelner Datensätze erschwert
- Aggregationslogik muss gepflegt und konsistent mit dem Quelldatenmodell gehalten werden
- Fehler in der Aggregation können unentdeckt bleiben und irreführende Berichte erzeugen
- Das Ändern von Aggregationsdimensionen im Nachhinein erfordert die Neuverarbeitung historischer Daten

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-IoT-Überwachungsplattform speicherte einzelne Sensorwerte jede Sekunde und häufte Milliarden von Zeilen pro Monat an. Dashboard-Abfragen, die stündliche Durchschnitte über das vergangene Jahr berechneten, brauchten über zwei Minuten, was die Anwendung für Betriebsteams nahezu unbenutzbar machte. Das Team führte eine Aggregationspipeline ein, die stündliche und tägliche Zusammenfassungen berechnete, während neue Daten eintrafen. Dashboard-Abfragen lesen jetzt aus den aggregierten Tabellen und liefern Ergebnisse in unter einer Sekunde. Die rohen sekundengenauen Daten wurden 90 Tage für detaillierte Fehlersuche aufbewahrt, während aggregierte Daten unbegrenzt behalten wurden, was die aktive Datensatzgröße um über 95 Prozent reduzierte.
