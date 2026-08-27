---
title: Read Replicas
description: Verteilung der Abfragelast auf schreibgeschützte
  Datenbank-Replikate abseits des Primärsystems.
category:
- Database
- Performance
problems:
- slow-database-queries
- high-database-resource-utilization
- scaling-inefficiencies
- database-query-performance-issues
- bottleneck-formation
- single-points-of-failure
- lock-contention
layout: solution
lang: de
en_slug: read-replicas
related_solutions:
- slug: data-replication
  similarity: 0.9
- slug: denormalization
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.8
- slug: data-partitioning
  similarity: 0.75
- slug: materialized-views
  similarity: 0.75
- slug: load-balancing
  similarity: 0.75
---

## Description

Read Replicas sind schreibgeschützte Kopien einer primären Datenbank, aktuell gehalten durch die eingebaute Replikation der Datenbank-Engine, an die die Leseabfragen einer Anwendung geleitet werden, während Schreibvorgänge weiterhin an den Primärserver gehen — entweder durch Änderungen an der Datenzugriffsschicht oder transparent über einen Verbindungsproxy für Legacy-Anwendungen, die sich nicht leicht modifizieren lassen. Dies ist ein häufiger und vergleichsweise wenig disruptiver Weg, die Datenbankschicht eines Legacy-Systems zu skalieren, weil es keine Änderung am Schema oder Grunddatenmodell erfordert, nur eine Routing-Entscheidung darüber, welche Abfragen wohin gehen, was es sogar für Systeme machbar macht, deren Kernlogik zu riskant oder schlecht verstanden ist, um direkt refaktoriert zu werden. Es ist besonders effektiv in Legacy-Systemen, die eine einzelne Datenbankinstanz sowohl für transaktionalen Anwendungs-Traffic als auch schwerere analytische oder Reporting-Abfragen nebeneinander haben wachsen lassen, da diese Reporting-Abfragen genau die Art lesehäufiger, latenztoleranter Arbeitslast sind, die vollständig vom Primärserver verschoben werden kann, was die Lock Contention beseitigt, die sie gegen transaktionale Schreibvorgänge verursachen. Die unvermeidbare Konsequenz asynchroner Replikation ist Replikationsverzögerung, was bedeutet, dass Replikate kurzzeitig veraltete Daten liefern können, sodass jeder Legacy-Workflow, der davon abhängt, sofort seinen eigenen gerade geschriebenen Wert zu lesen, identifiziert und explizit zum Primärserver statt zu einem Replikat geleitet werden muss. Über diese Konsistenzeinschränkung hinaus fügt jedes Replikat laufende Infrastruktur- und Betriebskosten hinzu, und Schemaänderungen müssen nun über den Primärserver und jedes Replikat hinweg koordiniert und konsistent angewendet werden statt an einer einzelnen Datenbankinstanz.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Analysieren Sie das Lese-/Schreib-Verhältnis des Datenbank-Traffics, um zu bestimmen, wie viel Last auf Replikate ausgelagert werden kann
- Richten Sie ein oder mehrere Read Replicas mittels der eingebauten Replikationsfeatures der Datenbank-Engine ein
- Modifizieren Sie die Datenzugriffsschicht, um Leseabfragen an Replikate und Schreibabfragen an den Primärserver zu leiten
- Nutzen Sie einen Verbindungsproxy oder Middleware, um Lese-/Schreib-Aufteilung transparent zu handhaben, wenn die Legacy-Anwendung nicht leicht modifiziert werden kann
- Berücksichtigen Sie Replikationsverzögerung in der Anwendungslogik und stellen Sie sicher, dass Operationen, die Read-Your-Writes-Konsistenz erfordern, den Primärserver nutzen
- Überwachen Sie Replikationsverzögerung und Replikat-Gesundheit kontinuierlich, mit Alarmen für inakzeptable Verzögerungen
- Beginnen Sie mit Reporting- und Analytics-Abfragen auf Replikaten, bevor Sie transaktionalen Lese-Traffic verschieben

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert Last auf der primären Datenbank und verbessert Schreibperformance und Gesamtstabilität
- Bietet einen skalierbaren Pfad für lesehäufige Arbeitslasten ohne Anwendungsneugestaltung
- Read Replicas können als warme Standbys für Disaster Recovery dienen
- Ermöglicht die Ausführung teurer Berichte und Analysen, ohne Produktionsperformance zu beeinträchtigen

**Kosten und Risiken:**
- Replikationsverzögerung bedeutet, dass Replikate leicht veraltete Daten liefern können
- Legacy-Anwendungen mit eng gekoppelten Read-after-Write-Mustern erfordern sorgfältiges Refactoring
- Jedes Replikat fügt Infrastruktur- und Betriebskosten hinzu
- Failover-Logik zwischen Primärserver und Replikaten fügt Komplexität hinzu
- Schemaänderungen müssen über Primärserver und alle Replikate koordiniert werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Die einzelne PostgreSQL-Instanz einer Legacy-E-Commerce-Plattform bediente sowohl transaktionalen Traffic als auch Business-Intelligence-Abfragen. Während Verkaufsereignissen verursachten Analytics-Abfragen des BI-Teams Lock Contention, die Checkout-Operationen verlangsamte. Das Team stellte zwei Read Replicas bereit: eines dediziert für die BI-Werkzeuge und ein anderes für die lesehäufigen API-Endpunkte des Produktkatalogs. Ein Verbindungsproxy leitete Abfragen transparent basierend auf der anfragenden Anwendung. Dies reduzierte die CPU-Auslastung der primären Datenbank während Spitzenereignissen um 60 Prozent und beseitigte die Interferenz zwischen Analytics und Transaktionsverarbeitung vollständig.
