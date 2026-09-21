---
title: Selbstüberwachung und Diagnose
description: Fähigkeit eines Systems, seinen eigenen Zustand zu überwachen
  und Probleme zu erkennen.
category:
- Operations
- Architecture
problems:
- monitoring-gaps
- slow-incident-resolution
- unpredictable-system-behavior
- gradual-performance-degradation
- constant-firefighting
- system-outages
layout: solution
lang: de
en_slug: self-monitoring-and-diagnosis
related_solutions:
- slug: self-test
  similarity: 0.8
- slug: monitoring
  similarity: 0.8
- slug: status-monitoring
  similarity: 0.75
- slug: watchdog
  similarity: 0.75
- slug: logging
  similarity: 0.75
- slug: heartbeat
  similarity: 0.7
---

## Description

Selbstüberwachung und Diagnose bettet Health Checks und interne Konsistenzverifikation direkt in eine Komponente ein, sodass sie ihre eigenen Ressourcenlecks, Dateninkonsistenzen und Logikfehler aus ihrem eigenen Ausführungskontext heraus erkennen kann, statt sich vollständig auf externes Monitoring zu verlassen, das Symptome nur von außen beobachten kann. Diese Unterscheidung zählt speziell für Legacy-Systeme, weil viele der subtilsten Fehlermodi — ein Hintergrund-Thread, der still bei einer fehlgeformten Eingabe stirbt, eine langsame Anhäufung einer internen Invariantenverletzung — überhaupt kein extern sichtbares Signal produzieren, bis der Fehler bereits Schaden verursacht hat, und externe Gesundheitsmetriken die ganze Zeit über völlig normal aussehen können. Diagnose-Endpunkte und strukturiertes Logging interner Befunde machen diese sonst unsichtbaren Probleme umsetzbar, und das Paaren von Erkennung mit automatischer Behebung für bekannte Muster, wie das Leeren eines Caches oder den Neustart eines hängenden Threads, verwandelt Diagnose in Selbstheilung. Der Selbstüberwachungscode selbst muss korrekt und leichtgewichtig sein, da fehlerhafte Diagnoselogik Fehlalarme produzieren oder eigenen Overhead hinzufügen kann.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Betten Sie Diagnosefähigkeiten in Legacy-Komponenten ein, die kontinuierlich ihre eigene betriebliche Gesundheit prüfen
- Implementieren Sie interne Konsistenzprüfungen, die Dateninvarianten und Verarbeitungskorrektheit verifizieren
- Fügen Sie automatische Erkennung von Ressourcenlecks (Speicher, Verbindungen, Datei-Handles) innerhalb der Anwendung hinzu
- Erstellen Sie Diagnose-Endpunkte, die internen Zustand für Fehlersuche ohne externes Tooling offenlegen
- Implementieren Sie automatische Behebung für bekannte, selbst diagnostizierbare Probleme (Connection-Pool-Auffrischung, Cache-Leerung)
- Protokollieren Sie Diagnosebefunde mit strukturierten Daten, um automatisierte Analyse und Alarmierung zu ermöglichen
- Gestalten Sie Selbstüberwachung so, dass sie elegant degradiert, sodass Monitoring-Fehler die Kernfunktionalität nicht beeinträchtigen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht schnellere Problemerkennung durch Überwachung aus dem eigenen Kontext der Anwendung heraus
- Erfasst interne Probleme, die externes Monitoring nicht beobachten kann (Logikfehler, Dateninkonsistenzen)
- Reduziert die Abhängigkeit von externer Monitoring-Infrastruktur
- Kann automatisierte Selbstheilung für bekannte Problemmuster auslösen

**Kosten und Risiken:**
- Selbstüberwachungscode fügt Komplexität hinzu und muss selbst korrekt sein, um Fehldiagnosen zu vermeiden
- Monitoring-Overhead im Anwendungsprozess kann die Performance beeinträchtigen
- Selbstüberwachung hat blinde Flecken für Probleme, die den Monitoring-Code selbst betreffen
- Legacy-Systemen könnten Erweiterungspunkte für das Hinzufügen interner Überwachung fehlen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Zahlungsverarbeitungssystem erlebte intermittierende Fehler, die externes Monitoring nicht erklären konnte, weil alle Gesundheitsmetriken normal erschienen. Durch das Hinzufügen von Selbstüberwachung, die interne Warteschlangentiefen, Transaktionsverarbeitungsraten und Datenkonsistenzprüfsummen verfolgte, erkannte das System ein subtiles Problem, bei dem ein Hintergrund-Thread still starb, nachdem er einen spezifischen fehlgeformten Nachrichtentyp verarbeitet hatte. Das Selbstüberwachungssystem startete den Thread automatisch neu und protokollierte die problematische Nachricht zur Untersuchung, was Zahlungsverarbeitungsverzögerungen verhinderte, die zuvor stundenlang unentdeckt geblieben waren.
