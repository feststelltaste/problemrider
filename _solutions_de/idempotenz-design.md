---
title: Idempotenz-Design
description: Gestaltung sicher wiederholbarer Operationen ohne unbeabsichtigte
  Nebeneffekte.
category:
- Architecture
- Code
problems:
- cascade-failures
- silent-data-corruption
- unpredictable-system-behavior
- inadequate-error-handling
- data-migration-integrity-issues
- race-conditions
- deadlock-conditions
layout: solution
lang: de
en_slug: idempotency-design
related_solutions:
- slug: idempotent-operations
  similarity: 0.95
- slug: transactions
  similarity: 0.75
- slug: retry
  similarity: 0.75
- slug: saga-pattern
  similarity: 0.7
- slug: redundancy
  similarity: 0.65
- slug: batch-processing
  similarity: 0.65
---

## Description

Idempotenz-Design ist die Praxis, eine Operation bewusst so zu konstruieren, dass ihre mehrfache Ausführung denselben Effekt hat wie eine einmalige, meist indem jeder Anfrage ein eindeutiger Idempotenzschlüssel zugewiesen wird, das Ergebnis der ersten Ausführung gegen diesen Schlüssel gespeichert wird und bei jedem nachfolgenden Retry das gespeicherte Ergebnis zurückgegeben wird, statt den zugrundeliegenden Nebeneffekt zu wiederholen. Als Designdisziplin wird sie an dem Punkt angewendet, an dem neue oder geänderte Operationen gebaut oder nachgerüstet werden, wobei fallweise entschieden wird, welche Operationen zu absoluter, Upsert-artiger Semantik umgewandelt werden können und welche von Natur aus destruktiv sind und eine alternative Absicherung wie explizite Deduplizierung brauchen. Dies ist in Legacy-Systemen besonders folgenreich, weil ihre zustandsverändernden Operationen häufig als blinde Inserts oder Inkremente geschrieben wurden, lange bevor Netzwerk-Retries, Nachrichtenwiederzustellung oder verteilte Fehlermodi als ständiges Designanliegen behandelt wurden, was doppelte Belastungen, doppelte Datensätze oder doppelt gezählte Inkremente als latentes Risiko hinterlässt, wann immer ein Client ein Timeout erleidet und erneut einreicht. Idempotenz in eine solche Operation nachzurüsten erfordert eine sorgfältige Prüfung ihrer Nebeneffekte — zu entscheiden, zum Beispiel, ob eine doppelte Belastung oder eine doppelte E-Mail der tolerierbarere Fehler ist, während die Behebung unvollständig ist —, da nicht jede Legacy-Operation günstig oder sicher idempotent gemacht werden kann. Sobald Idempotenzschlüssel und zwischengespeicherte Ergebnisse vorhanden sind, gewinnen Aufrufer die Freiheit, aggressiv zu wiederholen, ohne Angst vor Nebeneffekten, was brüchige manuelle Fehlerbehandlung und Umkehrverfahren durch ein wesentlich einfacheres und automatisierbareres Wiederherstellungsmodell ersetzt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie alle Operationen im Legacy-System, die Zustand ändern, und bewerten Sie, welche sicher wiederholt werden können
- Weisen Sie Anfragen eindeutige Idempotenzschlüssel zu, damit doppelte Einreichungen dasselbe Ergebnis produzieren
- Speichern Sie das Ergebnis abgeschlossener Operationen unter ihrem Idempotenz-Token, um bei Retry zwischengespeicherte Antworten zurückzugeben
- Wandeln Sie destruktive Operationen (Inkrementieren, Anhängen) wo möglich in absolute Operationen (auf Wert setzen) um
- Fügen Sie Deduplizierungsprüfungen an Diensteinstiegspunkten hinzu, um doppelte Nachrichten zu erkennen und zu verwerfen
- Gestalten Sie Datenbankoperationen mit Upsert-Semantik statt blinder Inserts
- Dokumentieren Sie, welche API-Endpunkte und Nachrichten-Handler idempotent sind und welche nicht

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht sichere Retry-Logik, die sich automatisch von vorübergehenden Fehlern erholt
- Verhindert doppelte Transaktionen, Belastungen oder Dateneinträge durch Netzwerk-Timeouts
- Vereinfacht Fehlerbehandlung, da Aufrufer sicher wiederholen können, ohne Nebeneffekte zu fürchten
- Unterstützt zuverlässige Nachrichtenverarbeitung in verteilten Legacy-Systemen

**Kosten und Risiken:**
- Erfordert zusätzlichen Speicher für Idempotenzschlüssel und zwischengespeicherte Ergebnisse
- Das Nachrüsten von Idempotenz in bestehende Operationen erfordert sorgfältige Analyse der Nebeneffekte
- Schlüsselablaufrichtlinien müssen Speicherkosten gegen Anforderungen des Retry-Fensters abwägen
- Manche Operationen sind von Natur aus nicht idempotent und brauchen alternative Strategien

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Zahlungsverarbeitungssystem erzeugte gelegentlich doppelte Belastungen, wenn Netzwerk-Timeouts den Client zu erneuten Einreichungen veranlassten. Das Team fügte Zahlungsanfragen Idempotenzschlüssel hinzu und speicherte abgeschlossene Transaktionsergebnisse in einer Deduplizierungstabelle. Wenn ein Retry mit demselben Schlüssel eintraf, gab das System das ursprüngliche Ergebnis zurück, ohne die Zahlung erneut zu verarbeiten. Dies beseitigte Beschwerden über doppelte Belastungen und erlaubte dem Team, aggressive Retry-Logik zum Client hinzuzufügen, was die Gesamtzuverlässigkeit verbesserte, ohne finanzielle Fehler zu riskieren.
