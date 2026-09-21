---
title: Redundante Datenspeicherung
description: Speicherung von Daten auf mehreren Medien oder Systemen.
category:
- Database
- Operations
problems:
- single-points-of-failure
- silent-data-corruption
- system-outages
- data-migration-integrity-issues
- unbounded-data-growth
layout: solution
lang: de
en_slug: redundant-data-storage
related_solutions:
- slug: redundancy
  similarity: 0.85
- slug: data-replication
  similarity: 0.85
- slug: regular-backups
  similarity: 0.8
- slug: data-deduplication
  similarity: 0.8
- slug: data-archiving
  similarity: 0.8
- slug: platform-independent-data-storage
  similarity: 0.8
---

## Description

Redundante Datenspeicherung bedeutet, mehrere Kopien desselben Datensatzes über verschiedene Festplatten, Speichersysteme oder geografische Standorte hinweg zu pflegen, sodass der Verlust oder die Korruption einer einzelnen Kopie nicht den Verlust der Daten selbst bedeutet. Implementierungen reichen von lokalen RAID-Arrays und Datenbankreplikation bis zu regionsübergreifenden Replikaten und Offsite-Archiven und können synchron sein — was null Datenverlust garantiert, aber Schreiblatenz hinzufügt — oder asynchron, was schneller ist, aber riskiert, die aktuellsten Transaktionen während eines Ausfalls zu verlieren. Legacy-Systeme laufen häufig auf alternder, Single-Instance-Speicherhardware, die noch nie ausgefallen ist, was genau der Grund ist, warum ihr eventueller Ausfall tendenziell katastrophal ist: Ohne eine zweite Kopie kann ein einzelner Festplatten- oder Controllerfehler Jahre an Geschäftsdaten löschen, die jede moderne Backup-Disziplin vordatieren. Die Einführung redundanter Speicherung ist oft einer der ersten Schritte zur Risikominderung in einer Legacy-Umgebung, bevor tiefere Modernisierungsarbeit beginnt, weil sie einen Hardwareausfall von einem existenzbedrohenden Datenverlustereignis in ein routinemäßiges Failover verwandelt. Sie schafft auch eine Grundlage, auf der spätere Modernisierungsaufwände — wie die Migration zu Cloud-Speicher oder die Aufteilung einer monolithischen Datenbank — aufbauen können, da ein System, das bereits an replizierten Speicher gewöhnt ist, leichter schrittweise migriert werden kann, ohne einen einzelnen Umschaltmoment maximalen Risikos.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Implementieren Sie Datenbankreplikation, um Kopien kritischer Daten auf separaten Speichersystemen zu pflegen
- Nutzen Sie RAID-Konfigurationen für lokalen Speicher, um vor einzelnen Festplattenausfällen zu schützen
- Replizieren Sie Daten über geografische Standorte hinweg für Disaster-Recovery-Szenarien
- Wählen Sie angemessene Replikationsstrategien: synchron für null Datenverlust, asynchron für Performance
- Verifizieren Sie Datenkonsistenz zwischen Replikaten regelmäßig mittels automatisierter Vergleichswerkzeuge
- Gestalten Sie die Anwendung so, dass sie von Replikaten liest, um Last zu verteilen und automatisches Failover zu bieten
- Dokumentieren Sie Wiederherstellungsverfahren für den Wechsel zu Replikatdaten, wenn der primäre Speicher ausfällt

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Schützt vor Datenverlust durch Hardwareausfälle, Korruption oder Katastrophen
- Ermöglicht fortgesetzten Betrieb, wenn primärer Speicher nicht verfügbar wird
- Read Replicas können die Abfrageperformance für stark belastete Legacy-Systeme verbessern
- Bietet eine Grundlage für Disaster Recovery und Business Continuity

**Kosten und Risiken:**
- Speicher- und Infrastrukturkosten vervielfachen sich mit jeder zusätzlichen Kopie
- Replikationsverzögerung in asynchronen Setups kann veraltete Lesevorgänge oder Datenkonflikte verursachen
- Die Verwaltung der Replikationstopologie fügt betriebliche Komplexität hinzu
- Legacy-Anwendungen handhaben Lese-/Schreib-Aufteilung möglicherweise nicht ohne Modifikation

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Dokumentenverwaltungssystem einer Anwaltskanzlei speicherte alle Mandantendokumente auf einem einzelnen NAS-Gerät. Als das Gerät einen Controller-Ausfall erlebte, verlor die Kanzlei zwei Geschäftstage lang den Zugang zu Dokumenten, während auf Ersatzteile gewartet wurde. Nach der Wiederherstellung implementierte das Team redundante Speicherung mit Echtzeit-Replikation zu einem sekundären NAS und nächtlicher Replikation zu Cloud-Objektspeicher. Der nächste Hardwareausfall löste automatisches Failover zum sekundären NAS innerhalb von Minuten aus, und die Cloud-Kopie bot ein zusätzliches Sicherheitsnetz für Katastrophenszenarien.
