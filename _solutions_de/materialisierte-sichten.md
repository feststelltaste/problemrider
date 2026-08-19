---
title: Materialisierte Sichten
description: Optimierung der Datenbankabfrageperformance durch Speicherung von
  Abfrageergebnissen.
category:
- Database
- Performance
problems:
- slow-database-queries
- database-query-performance-issues
- high-number-of-database-queries
- high-database-resource-utilization
- slow-response-times-for-lists
- gradual-performance-degradation
- imperative-data-fetching-logic
- lazy-loading
- poor-caching-strategy
- entity-attribute-value-overuse
- custom-report-sprawl
layout: solution
lang: de
en_slug: materialized-views
related_solutions:
- slug: denormalization
  similarity: 0.85
- slug: query-optimization-process
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.8
- slug: data-partitioning
  similarity: 0.8
- slug: data-replication
  similarity: 0.8
- slug: data-archiving
  similarity: 0.75
---

## Description

Eine materialisierte Sicht berechnet das Ergebnis einer Abfrage vor und speichert es physisch — typischerweise einer, die große Tabellen aggregiert oder verbindet —, sodass nachfolgende Lesevorgänge dagegen sofort aus dem gespeicherten Ergebnis zurückkehren, statt bei jedem Zugriff die zugrundeliegenden Joins und Aggregationen neu zu berechnen, auf Kosten dessen, dass das gespeicherte Ergebnis veraltet, bis es als Nächstes nach einem periodischen, bedarfsgesteuerten oder inkrementellen Zeitplan aktualisiert wird. Dies ist eine Optimierung auf Datenbankebene, die oft eingeführt werden kann, ohne den Legacy-Anwendungscode überhaupt anzufassen, da bestehende Abfragen einfach zur materialisierten Sicht statt zu den Basistabellen umgeleitet werden können. Legacy-Systeme enthalten häufig langlebige Reporting- und Dashboard-Abfragen, die fünf oder mehr Tabellen verbinden und schnell genug waren, als das System klein war, aber schrittweise verkommen sind, während das Datenvolumen Jahr für Jahr wuchs, bis eine Abfrage, die einst Millisekunden brauchte, nun Zehntausende Millisekunden braucht und während der Geschäftszeiten einen unverhältnismäßigen Anteil der Datenbankkapazität verbraucht. Das Ergebnis einer solchen Abfrage zu materialisieren verschiebt die Rechenkosten von jedem Lesevorgang auf eine geplante Aktualisierung, was ein günstiger Tausch ist, wann immer die zugrundeliegenden Daten nicht in Echtzeit widergespiegelt werden müssen, und diese Veraltungstoleranz ist genau die Art von Zielkonflikt, der explizit gemacht und dokumentiert werden muss, damit nachgelagerte Konsumenten verstehen, welche Aktualitätsgarantie sie tatsächlich erhalten. Weil materialisierte Sichten Speicher und Aktualisierungsplan-Pflege zusätzlich zum Schema hinzufügen, fügen sie auch eine kleine Menge Fläche zu künftigen Schema-Migrationen hinzu, was Kosten sind, die gegen den gewonnenen Performance-Vorteil abzuwägen sind.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie teure, häufig ausgeführte Abfragen, die große Tabellen aggregieren oder verbinden und Ergebnisse produzieren, die sich selten ändern
- Erstellen Sie materialisierte Sichten, die die Ergebnisse dieser Abfragen vorberechnen und speichern
- Etablieren Sie eine Aktualisierungsstrategie (periodisch, bedarfsgesteuert oder inkrementell), die Datenaktualität gegen Ressourcenkosten abwägt
- Leiten Sie Anwendungsabfragen zu den materialisierten Sichten statt zu den Basistabellen um
- Fügen Sie Indizes auf den materialisierten Sichten hinzu, um nachgelagerte Abfragen weiter zu beschleunigen
- Überwachen Sie Aktualisierungszeiten und Speicherverbrauch, um sicherzustellen, dass die materialisierten Sichten nachhaltig bleiben, während Daten wachsen
- Dokumentieren Sie die Veraltungstoleranz für jede Sicht, damit das Team die Aktualitätsgarantien versteht

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert die Ausführungszeit komplexer Aggregationen und Joins dramatisch
- Entlastet die primäre Datenbank während Spitzenzeiten, wenn Aktualisierungen außerhalb der Spitzenzeiten geplant sind
- Kann ohne Änderung des Legacy-Anwendungscodes eingeführt werden, wenn Abfragen auf Datenbankebene umgeleitet werden
- Verringert Konkurrenz auf stark abgefragten Tabellen

**Kosten und Risiken:**
- Materialisierte Sichten verbrauchen zusätzlichen Speicher und erfordern Pflege des Aktualisierungsplans
- Veraltete Daten zwischen Aktualisierungen können falsche Ergebnisse verursachen, wenn die Veraltungstoleranz nicht gut verstanden ist
- Aktualisierungsoperationen selbst können ressourcenintensiv sein und müssen sorgfältig geplant werden
- Das Hinzufügen materialisierter Sichten erhöht die Fläche von Schemaänderungen während Migrationen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Reporting-Dashboard eines Versicherungsunternehmens verband fünf große Tabellen, um Schadenzusammenfassungen zu berechnen, was während der Geschäftszeiten über 30 Sekunden pro Abfrage brauchte. Das Team erstellte eine materialisierte Sicht, die die Zusammenfassung vorberechnete und alle 15 Minuten aktualisierte. Die Antwortzeiten des Dashboards sanken auf unter eine Sekunde, und die Datenbank-CPU-Auslastung während Spitzenzeiten fiel um 40 Prozent. Die 15-minütige Veraltung war für den Reporting-Anwendungsfall akzeptabel, und das Team dokumentierte diese Einschränkung, damit Echtzeit-Datenbedürfnisse an einen anderen Abfragepfad geleitet würden.
