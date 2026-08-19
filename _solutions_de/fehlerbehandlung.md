---
title: Fehlerbehandlung
description: Mechanismen zur Erkennung, Protokollierung und Behandlung von Fehlern.
category:
- Code
- Architecture
problems:
- inadequate-error-handling
- cascade-failures
- unpredictable-system-behavior
- debugging-difficulties
- silent-data-corruption
- increased-error-rates
- slow-incident-resolution
- null-pointer-dereferences
- stack-overflow-errors
- unreleased-resources
- database-connection-leaks
- improper-event-listener-management
layout: solution
lang: de
en_slug: error-handling
related_solutions:
- slug: error-reporting-and-analysis
  similarity: 0.85
- slug: error-logging
  similarity: 0.85
- slug: exceptions
  similarity: 0.8
- slug: error-logs
  similarity: 0.8
- slug: logging
  similarity: 0.8
- slug: retry
  similarity: 0.75
---

## Description

Fehlerbehandlung umfasst die Mechanismen, mittels derer ein System erkennt, dass etwas schiefgelaufen ist, entscheidet, was dagegen zu tun ist, und das Ergebnis kommuniziert — schnell scheitern bei nicht wiederherstellbaren Zuständen, vorübergehende Fehler mit Backoff wiederholen oder graziös degradieren für nicht-kritische Funktionalität —, statt Verhalten an Fehlerpunkten undefiniert zu lassen. Legacy-Codebasen häufen hier ein bestimmtes Fehlermuster an: generische Catch-all-Blöcke, die etwas Vages wie „ein Fehler ist aufgetreten" protokollieren und die ursprüngliche Exception verschlucken, hinzugefügt über Jahre von Entwicklern, die wollten, dass die Anwendung weiterläuft statt abzustürzen, auf Kosten der Löschung der Information, die nötig ist, um das tatsächliche Problem später zu diagnostizieren. Diese Catch-alls durch spezifische, an unterschiedliche Fehlertypen gebundene Handler zu ersetzen, kontextuelle Informationen zu jedem Fehler hinzuzufügen und die Behandlungslogik an definierten Grenzen zu zentralisieren verwandelt Fehlerbehandlung von Rauschen in ein Diagnosewerkzeug, was genau das ist, was nötig ist, wenn die ursprünglichen Autoren und die Dokumentation eines Systems nicht mehr verfügbar sind, um die Lücken zu füllen. Weil das nachträgliche Einbauen in funktionierenden Legacy-Code viele Aufrufpfade gleichzeitig berührt, ist das Hauptrisiko, unbeabsichtigt beobachtbares Verhalten zu ändern, von dem nachgelagerte Konsumenten still abhängig geworden sind, sodass der Aufwand schrittweise erfolgen und mit ausreichender Testabdeckung gepaart werden muss.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Prüfen Sie die Codebasis auf verschluckte Exceptions, leere Catch-Blöcke und generische Fehlerbehandler, die Ursachen verbergen
- Etablieren Sie eine konsistente Fehlerbehandlungsstrategie: schnell scheitern bei nicht wiederherstellbaren Fehlern, mit Backoff wiederholen bei vorübergehenden Fehlern und graziös degradieren bei nicht-kritischen Features
- Ersetzen Sie Catch-all-Exception-Handler durch spezifische Handler, die für jeden Fehlertyp angemessene Maßnahmen ergreifen
- Fügen Sie Fehlermeldungen und Log-Einträgen kontextuelle Informationen hinzu, um Diagnose schneller zu machen
- Implementieren Sie strukturierte Fehlerantworten für APIs, die aussagekräftige Fehlercodes, Meldungen und vorgeschlagene Aktionen liefern
- Erstellen Sie zentralisierte Fehlerbehandlungs-Middleware, statt Try-Catch-Blöcke über die Codebasis zu verstreuen
- Fügen Sie Überwachung und Alarmierung für Fehlerraten hinzu, damit sich abzeichnende Probleme erkannt werden, bevor sie zu Ausfällen werden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Macht das System vorhersehbar, indem explizites Verhalten für jeden Fehlermodus definiert wird
- Verbessert die Debugging-Geschwindigkeit durch kontextuelle Fehlerinformationen
- Verhindert stille Fehler, die zu Datenkorruption oder inkonsistentem Zustand führen
- Ermöglicht schnellere Vorfalllösung durch klare Fehlersignale
- Reduziert kaskadierende Fehler, indem Fehler an angemessenen Grenzen eingedämmt werden

**Kosten und Risiken:**
- Fehlerbehandlung nachträglich in Legacy-Code einzubauen ist arbeitsintensiv und riskiert Verhaltensänderung
- Übermäßig aggressive Fehlerbehandlung (überall schnell scheitern) kann die Systemverfügbarkeit reduzieren
- Ausführliche Fehlermeldungen könnten unbeabsichtigt sensible Systemdetails offenlegen
- Konsistente Fehlerbehandlung erfordert Teamdisziplin und laufende Code-Review-Aufmerksamkeit

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-E-Commerce-System hatte ein Muster, alle Exceptions mit generischen Handlern abzufangen, die „Ein Fehler ist aufgetreten" protokollierten und HTTP 500 zurückgaben. Wenn Produktionsprobleme auftraten, verbrachte das Team Stunden damit, vage Log-Einträge mit Nutzermeldungen zu korrelieren. Eine systematische Prüfung fand 340 generische Catch-Blöcke. Das Team ersetzte sie über drei Monate durch spezifische Handler: Validierungsfehler gaben 400 mit Detail auf Feldebene zurück, Authentifizierungsfehler gaben 401 mit klaren Meldungen zurück, und unerwartete Fehler beinhalteten Korrelations-IDs, die Logs mit Nutzersitzungen verknüpften. Die mittlere Zeit zur Diagnose von Produktionsproblemen sank von vier Stunden auf 30 Minuten, und die Anzahl der als „unbekannter Fehler" kategorisierten Support-Tickets sank um 85 Prozent.
