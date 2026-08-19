---
title: Denormalisierung
description: Einführung kontrollierter Redundanz in Datenbankschemata für schnellere
  Lesezugriffe.
category:
- Database
- Performance
problems:
- slow-database-queries
- database-query-performance-issues
- high-number-of-database-queries
- n-plus-one-query-problem
- slow-response-times-for-lists
- lazy-loading
layout: solution
lang: de
en_slug: denormalization
related_solutions:
- slug: materialized-views
  similarity: 0.85
- slug: data-replication
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.8
- slug: read-replicas
  similarity: 0.8
- slug: data-partitioning
  similarity: 0.8
- slug: nosql-databases
  similarity: 0.75
---

## Description

Denormalisierung führt bewusst kontrollierte Redundanz in ein Datenbankschema ein — Duplizieren oder Vorberechnen von Werten, die sonst einen teuren Join oder eine Aggregation zur Abfragezeit erfordern würden — und tauscht Schreibzeit-Komplexität und zusätzlichen Speicher gegen dramatisch schnellere Lesevorgänge. In der Praxis bedeutet dies, berechnete oder gecachte Spalten für häufig benötigte abgeleitete Werte direkt zu den Tabellen hinzuzufügen, die Konsumenten abfragen, oder separate Zusammenfassungstabellen zu pflegen, und dann diese denormalisierten Werte mit ihrer Quelle der Wahrheit mittels Triggern, Anwendungsebenen-Hooks oder Event-Handlern synchron zu halten, statt darauf zu vertrauen, dass sie von selbst korrekt bleiben. Legacy-Systeme häufen genau die Bedingungen an, die dies lohnenswert machen: Schemata, die vor Jahrzehnten aus Datenintegritätsgründen normalisiert wurden, bedienen jetzt lesestarke Zugriffsmuster, die Joins über viele Tabellen hinweg erfordern, nur um eine einzige Seite zu rendern, und unter echter Produktionslast verwandeln diese Joins routinemäßig das, was eine schnelle Abfrage sein sollte, in eine mehrsekündige Abfrage. Denormalisierung selektiv anzuwenden, beginnend mit lesestarken und schreibarmen Bereichen, erlaubt einem Team, die schlimmsten Abfragen zu eliminieren, ohne das Schema als Ganzes umzustrukturieren, während die Dokumentation, welche Quelltabellen maßgeblich bleiben, verhindert, dass die Redundanz zu einem unüberschaubaren Geflecht konkurrierender „Wahrheiten" wird. Weil jeder denormalisierte Wert über die Zeit von seiner Quelle abdriften kann, wegen eines verpassten Aktualisierungspfads oder eines Fehlers in einem Synchronisations-Hook, erfordert das Muster laufende Abgleichsprüfungen, um Inkonsistenzen zu erkennen und zu korrigieren, bevor sie für maßgebliche Daten gehalten werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie die teuersten Abfragen, die mehrere Tabellen verbinden, und analysieren Sie, ob vorab verbundene Daten den Engpass eliminieren würden
- Fügen Sie berechnete oder gecachte Spalten hinzu, die häufig benötigte abgeleitete Werte speichern (z. B. Bestellsummen, Anzeigenamen)
- Erstellen Sie Zusammenfassungstabellen, die aggregierte Daten für schnellen Abruf duplizieren
- Implementieren Sie Trigger, Anwendungsebenen-Hooks oder Event-Handler, um denormalisierte Daten mit Quelldaten synchron zu halten
- Dokumentieren Sie jede Denormalisierungsentscheidung einschließlich, welche Quelltabellen maßgeblich sind
- Beginnen Sie mit lesestarken, schreibarmen Bereichen, wo der Synchronisationsoverhead minimal ist
- Überwachen Sie auf Dateninkonsistenzen zwischen normalisierten und denormalisierten Kopien

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Eliminiert teure Joins und Aggregationen zur Abfragezeit durch Vorberechnung von Ergebnissen
- Verbessert die Leseperformance für komplexe Abfragen dramatisch
- Reduziert die Datenbanklast, indem wiederholte Berechnung derselben abgeleiteten Daten vermieden wird
- Kann selektiv angewandt werden, ohne das gesamte Schema umzustrukturieren

**Kosten und Risiken:**
- Führt Datenredundanz ein, die synchron gehalten werden muss, was Inkonsistenzen riskiert
- Schreiboperationen werden komplexer und potenziell langsamer wegen Synchronisationsoverhead
- Speicheranforderungen steigen durch duplizierte Daten
- Die Schemakomplexität wächst mit zusätzlichen Spalten und Tabellen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-E-Commerce-Plattform hatte eine Produktlistenseite, die sieben Tabellen verbinden musste, um jedes Produkt mit seinem Kategorienamen, durchschnittlicher Bewertung, aktuellem Preis und Bestandsstatus anzuzeigen. Unter Last dauerte diese Abfrage über zwei Sekunden für eine Seite mit 50 Produkten. Das Team fügte der Produkttabelle direkt denormalisierte Spalten hinzu: `category_name`, `avg_rating`, `current_price` und `stock_status`. Anwendungsebenen-Event-Listener aktualisierten diese Spalten, wann immer sich die Quelldaten änderten. Die Produktlistenabfrage wurde zu einem Einzeltabellen-Scan, der in unter 50 Millisekunden zurückkam. Das Team fügte einen nächtlichen Abgleichsjob hinzu, um jede Drift zwischen den denormalisierten Spalten und ihren Quelltabellen zu erkennen und zu korrigieren.
