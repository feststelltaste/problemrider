---
title: Datenreplikation
description: Erstellung und Synchronisierung von Datenkopien über mehrere Systeme
  hinweg.
category:
- Database
- Architecture
problems:
- single-points-of-failure
- system-outages
- cross-system-data-synchronization-problems
- slow-database-queries
- high-database-resource-utilization
- scaling-inefficiencies
layout: solution
lang: de
en_slug: data-replication
related_solutions:
- slug: read-replicas
  similarity: 0.9
- slug: redundant-data-storage
  similarity: 0.85
- slug: distributed-caching
  similarity: 0.8
- slug: denormalization
  similarity: 0.8
- slug: distributed-processing
  similarity: 0.8
- slug: redundancy
  similarity: 0.8
---

## Description

Datenreplikation erstellt und synchronisiert kontinuierlich Kopien eines Datensatzes über mehrere Systeme oder Knoten hinweg, mittels synchroner oder asynchroner Mechanismen je nach Strenge der geforderten Konsistenz, sodass dieselben Daten von mehr als einem Ort für Lesen, Failover oder geografische Verteilung verfügbar sind. In der Praxis bedeutet dies meist, ein primäres System of Record und eine oder mehrere Replikate zu bestimmen, die einen kontinuierlichen Strom von Änderungen erhalten — entweder durch native Datenbankreplikation oder durch Change Data Capture, das den Primärknoten beobachtet, ohne Modifikationen an ihm zu erfordern —, mit Überwachung, um Replikationslag oder Synchronisationsfehler zu erkennen. Für Legacy-Systeme adressiert Replikation zwei unterschiedliche Schmerzpunkte gleichzeitig: Eine einzelne Datenbankinstanz, die sowohl transaktionale als auch Berichts-Workloads bedienen muss, leidet unter Lock Contention und Verlangsamungen, wenn beide um dieselben Ressourcen konkurrieren, und eine einzelne Datenbankinstanz ohne Standby ist auch ein Single Point of Failure, der jedes Hardwareproblem in einen längeren Ausfall verwandelt. Lesestarken Berichtstraffic zu Replikaten zu leiten entlastet den Primärknoten, während ein geografisch getrenntes Replikat gleichzeitig als Disaster-Recovery-Ziel dient, das befördert werden kann, falls der Primärknoten nicht mehr verfügbar ist. Der Tradeoff, der Replikation inhärent ist, ist, dass Kopien nicht sofort konsistent sind — Replikationslag kann veraltete Lesevorgänge erzeugen, und jede Konfiguration, die Schreibvorgänge auf mehr als eine Kopie erlaubt, führt Konflikte ein, die durch eine explizite Strategie gelöst werden müssen, statt dem Zufall überlassen zu werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Bewerten Sie die Replikationsfähigkeiten der Legacy-Datenbank und bestimmen Sie, ob synchrone oder asynchrone Replikation angemessen ist
- Richten Sie Lese-Replikate ein, um Berichts- und Analytics-Abfragen von der primären Datenbank zu entlasten
- Konfigurieren Sie Replikationsüberwachung, um Lag, Konflikte und Synchronisationsfehler zu erkennen
- Definieren Sie ein klares Konsistenzmodell (eventual, strong oder session consistency) basierend auf Geschäftsanforderungen
- Implementieren Sie Failover-Prozeduren, die ein Replikat zum Primärknoten befördern, wenn der Primärknoten nicht verfügbar wird
- Nutzen Sie Change Data Capture (CDC), um Daten an nachgelagerte Systeme zu replizieren, ohne die Legacy-Anwendung zu modifizieren
- Testen Sie Failover- und Wiederherstellungsprozeduren regelmäßig, um sicherzustellen, dass sie bei Bedarf funktionieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Eliminiert die Datenbank als Single Point of Failure durch Redundanz
- Verbessert die Leseperformance durch Verteilung von Abfragen über Replikate
- Ermöglicht geografische Datenverteilung zur Latenzreduzierung
- Unterstützt Disaster Recovery mit Off-Site-Datenkopien

**Kosten und Risiken:**
- Replikationslag kann veraltete Lesevorgänge und vorübergehende Inkonsistenzen verursachen
- Schreibkonflikte in Multi-Primary-Konfigurationen erfordern Konfliktlösungsstrategien
- Erhöht Speicher- und Infrastrukturkosten mit jedem zusätzlichen Replikat
- Die Überwachung und Verwaltung der Replikationsgesundheit fügt operative Komplexität hinzu
- Schemaänderungen müssen über alle Replikate hinweg sorgfältig koordiniert werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Bestandsverwaltungssystem lief auf einer einzelnen PostgreSQL-Datenbank, die sowohl transaktionaler Speicher als auch Quelle für Berichtsabfragen war. Umfangreiche Berichtsabfragen während der Geschäftszeiten verursachten Lock Contention, die die Auftragsverarbeitung verlangsamte. Das Team richtete zwei asynchrone Lese-Replikate ein und leitete alle Berichtsabfragen mittels einer Verbindungsrouting-Schicht dorthin. Die Transaktionsverarbeitungslatenz verbesserte sich während Spitzenzeiten um 40 Prozent. Zusätzlich wurde ein Replikat in einem sekundären Rechenzentrum platziert, was einen Warm Standby für Disaster Recovery bot. Als die primäre Datenbank sechs Monate später einen Hardwarefehler erlitt, führte das Team einen Failover zum Standby mit nur drei Minuten Datenverlust durch, verglichen mit den Stunden Ausfallzeit, die ohne Replikation entstanden wären.
