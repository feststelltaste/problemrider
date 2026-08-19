---
title: Fehlertolerante Datenstrukturen
description: Nutzung von Datenstrukturen, die trotz Fehlern oder Inkonsistenzen
  funktionsfähig bleiben.
category:
- Code
- Architecture
problems:
- silent-data-corruption
- inadequate-error-handling
- unpredictable-system-behavior
- brittle-codebase
- data-migration-integrity-issues
- cascade-failures
layout: solution
lang: de
en_slug: fault-tolerant-data-structures
related_solutions:
- slug: error-correction-codes
  similarity: 0.8
- slug: data-integrity
  similarity: 0.75
- slug: redundant-data-storage
  similarity: 0.75
- slug: checksums
  similarity: 0.75
- slug: retry
  similarity: 0.75
- slug: standardized-data-formats
  similarity: 0.75
---

## Description

Fehlertolerante Datenstrukturen sind darauf ausgelegt, Beschädigungen oder unvollständige Schreibvorgänge zu erkennen und, wo möglich, sich automatisch davon zu erholen, statt schlechte Daten still weiterzugeben oder gänzlich abzustürzen — durch Mechanismen wie Prüfsummen oder Versionsfelder, die in Datensätzen eingebettet sind, redundante oder selbstprüfende Strukturen wie integritätsgeprüfte B-Bäume und defensive Deserialisierung, die strukturelle Invarianten prüft, bevor eingehende Daten akzeptiert werden. Dies ist am wichtigsten für die kritischen, langlebigen Datenstrukturen im Kern von Legacy-Systemen, in denen Race Conditions, unvollständige Schreibvorgänge oder über Jahre angesammelte Formatdrift den Zustand auf Weisen beschädigen können, die unbemerkt bleiben, bis die Beschädigung bereits in nachgelagerte Berechnungen oder Berichte propagiert ist. Das Hinzufügen von Integritätsprüfung und Wiederherstellungslogik — der Fähigkeit, eine Struktur aus einem bekannt guten Zustand oder einem Protokoll neu aufzubauen oder zu reparieren — verwandelt zuvor stille, mysteriöse Datenprobleme in sichtbare, erkannte Ereignisse, und tut dies, ohne einen vollständigen Ersatz des umgebenden Legacy-Codes zu erfordern, der die Struktur liest und schreibt. Der Zielkonflikt sind zusätzlicher Speicher- und CPU-Overhead für die Redundanz und Validierung selbst, Migrationsaufwand zur Nachrüstung bestehender Datenformate und das Risiko, dass selbstheilendes Verhalten einen Nebenläufigkeits- oder Logikfehler übertüncht, der eigentlich an seiner Quelle behoben werden muss, statt fortlaufend um ihn herum korrigiert zu werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Prüfen Sie kritische Datenstrukturen in der Legacy-Codebasis auf Anfälligkeit für Beschädigung oder unvollständige Schreibvorgänge
- Führen Sie Prüfsummen oder Versionsfelder in Datensätzen ein, um Inkonsistenzen früh zu erkennen
- Nutzen Sie selbstheilende Datenstrukturen wie redundante verkettete Listen oder B-Bäume mit Integritätsprüfung
- Implementieren Sie defensive Deserialisierung, die strukturelle Invarianten prüft, bevor Daten akzeptiert werden
- Fügen Sie Wiederherstellungslogik hinzu, die Datenstrukturen aus bekannt gutem Zustand oder Protokollen neu aufbauen oder reparieren kann
- Umhüllen Sie Legacy-Datenzugriff mit Validierungsschichten, die beschädigte Einträge zurückweisen oder unter Quarantäne stellen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Das System arbeitet weiter, selbst wenn einzelne Datenelemente beschädigt sind
- Verringert stille Datenbeschädigung, die durch nachgelagerte Prozesse propagieren kann
- Macht Datenprobleme durch Integritätsprüfungen sichtbar statt durch mysteriöse Ausfälle
- Unterstützt sicherere Datenmigration durch Erkennung von Inkonsistenzen während des Übergangs

**Kosten und Risiken:**
- Fehlertolerante Strukturen nutzen mehr Speicher und CPU für Redundanz und Validierung
- Die Nachrüstung bestehender Datenformate erfordert sorgfältige Migrationsplanung
- Übermäßiges Vertrauen in Selbstheilung kann systemische Probleme verdecken, die eine Ursachenbehebung brauchen
- Zusätzliche Komplexität in Datenzugriffsschichten erhöht den Wartungsaufwand

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Telekommunikationsanbieter entdeckte, dass sein Legacy-Abrechnungssystem gelegentlich beschädigte Kundendatensätze aufgrund von Race Conditions in einer Shared-Memory-Datenstruktur erzeugte. Durch das Ersetzen des kritischen Kontostand-Caches durch eine versionierte Struktur mit CRC-Prüfungen und automatischem Rollback zum letzten gültigen Zustand beseitigte das Team Abrechnungsdiskrepanzen, die seit Jahren Kundenbeschwerden verursacht hatten. Die fehlertolerante Struktur protokollierte jedes erkannte Beschädigungsereignis, was dem Team auch half, den zugrundeliegenden Nebenläufigkeitsfehler zu identifizieren und zu beheben.
