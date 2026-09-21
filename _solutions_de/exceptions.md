---
title: Exceptions
description: Nutzung von Exceptions zur Signalisierung und Behandlung von Fehlerzuständen.
category:
- Code
problems:
- inadequate-error-handling
- debugging-difficulties
- unpredictable-system-behavior
- silent-data-corruption
- cascade-failures
- difficult-code-comprehension
layout: solution
lang: de
en_slug: exceptions
related_solutions:
- slug: error-handling
  similarity: 0.8
- slug: error-reporting-and-analysis
  similarity: 0.8
- slug: error-logging
  similarity: 0.75
- slug: pattern-language
  similarity: 0.7
- slug: error-logs
  similarity: 0.7
- slug: logging
  similarity: 0.7
---

## Description

Exceptions sind ein Sprachebenen-Mechanismus zur Signalisierung, dass eine Operation nicht wie erwartet abgeschlossen werden konnte, indem dieser Fehler den Call Stack hinaufgereicht wird, bis Code, der weiß, wie damit umzugehen ist, ihn abfängt, im Gegensatz zur Kodierung von Fehlern als Integer-Rückgabecodes, Boolean-Flags oder andere Werte, die ein Aufrufer still ignorieren kann. Viele Legacy-Codebasen, besonders solche mit Ursprung in C-artigen Sprachen, verlassen sich auf letzteren Ansatz, und weil nichts einen Aufrufer zwingt, einen Rückgabewert zu prüfen, sind ignorierte Fehlercodes ein häufiger Weg, auf dem ein Fehler an einer Stelle im Code still zu Datenkorruption oder einem Absturz an einer ganz anderen Stelle wird, ohne direkte Verbindung zwischen Ursache und schließlichem Symptom. Solchen Code zu einer typisierten Exception-Hierarchie zu migrieren, und diese Exceptions nur an wohldefinierten Grenzen wie der API-Schicht oder dem Einstiegspunkt eines Batch-Jobs abzufangen statt um jeden Aufruf herum, macht Fehlerzustände unmöglich zu übersehen und gibt dem Team einen Stack Trace und strukturierten Kontext, mit dem gearbeitet werden kann, wenn tatsächlich etwas schiefgeht. Der Übergang muss jedoch sorgfältig erfolgen, da die Einführung von Exceptions in Code, der zuvor auf Fehlercodes angewiesen war, beobachtbares Verhalten ändern kann, wenn nicht gründlich getestet, und in performance-sensiblen Pfaden auf manchen Plattformen sind die Kosten häufigen Werfens von Exceptions nicht trivial genug, dass sie für echt außergewöhnliche Bedingungen statt routinemäßigen Kontrollfluss reserviert werden sollten.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Ersetzen Sie Fehlercodes, Boolean-Rückgabewerte und stille Fehlschläge durch typisierte Exceptions, die klar beschreiben, was schiefgegangen ist
- Definieren Sie eine Hierarchie benutzerdefinierter Exception-Typen, die zwischen wiederherstellbaren und nicht wiederherstellbaren Fehlern unterscheidet
- Fangen Sie Exceptions an angemessenen Grenzen ab (Service-Schicht, API-Grenze, Batch-Job-Einstiegspunkt) statt bei jedem Methodenaufruf
- Verschlucken Sie Exceptions nie still; protokollieren, umhüllen oder werfen Sie immer mit zusätzlichem Kontext erneut
- Nutzen Sie Exception-Metadaten (Fehlercodes, betroffene Entitäten, vorgeschlagene Aktionen), um Aufrufern handlungsfähige Informationen zu liefern
- Etablieren Sie Teamkonventionen dafür, wann Checked- vs. Unchecked-Exceptions basierend auf Sprache und Framework genutzt werden
- Refaktorieren Sie Legacy-Code, der Fehlercodes oder magische Rückgabewerte nutzt, Modul für Modul, um stattdessen Exceptions zu werfen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Macht Fehlerzustände explizit und unmöglich zu ignorieren, anders als Rückgabecodes, die still verworfen werden können
- Trennt Fehlerbehandlungslogik vom normalen Fluss, was die Code-Lesbarkeit verbessert
- Bietet Stack-Trace-Kontext, der Debugging und Ursachenanalyse unterstützt
- Ermöglicht zentralisierte Fehlerbehandlung an architektonischen Grenzen
- Typisierte Exceptions erlauben Aufrufern, unterschiedliche Fehlerbedingungen spezifisch zu behandeln

**Kosten und Risiken:**
- Exceptions können in manchen Sprachen teuer sein (z. B. JVM-Stack-Trace-Erfassung), wenn häufig geworfen
- Übermäßige Nutzung von Exceptions für Kontrollfluss macht Code schwerer nachzuvollziehen und verschlechtert die Performance
- Nicht abgefangene Exceptions können die Anwendung zum Absturz bringen, wenn keine globalen Handler vorhanden sind
- Die Migration von Fehlercodes zu Exceptions in einer Legacy-Codebasis erfordert sorgfältiges Testing, um Verhalten zu bewahren
- Teams könnten uneinig sein, welche Bedingungen Exceptions vs. Rückgabewerte rechtfertigen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-C++-Anwendung nutzte von Funktionen zurückgegebene Integer-Fehlercodes, wobei -1 Fehlschlag bedeutete und verschiedene positive Werte spezifische Fehler anzeigten. Viele Aufrufstellen prüften Rückgabewerte nicht, was dazu führte, dass sich Fehler still fortpflanzten, bis sie sich als Datenkorruption oder Abstürze weit vom ursprünglichen Fehler entfernt manifestierten. Das Team führte eine benutzerdefinierte Exception-Hierarchie mit domänenspezifischen Typen wie InvalidOrderException und InsufficientInventoryException ein. Sie refaktorierten zuerst die kritischsten Module und umhüllten Legacy-Funktionen, die Fehlercodes zurückgaben, in Adapter-Funktionen, die Exceptions warfen. Innerhalb von vier Monaten sank die Anzahl der „mysteriösen Abstürze" um 70 Prozent, weil Fehler jetzt explizit nahe ihrer Quelle abgefangen und behandelt wurden.
