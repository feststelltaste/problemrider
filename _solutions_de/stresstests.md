---
title: Stresstests
description: Testen der Software unter extremen Lastbedingungen.
category:
- Testing
- Performance
problems:
- system-outages
- cascade-failures
- capacity-mismatch
- unpredictable-system-behavior
- scaling-inefficiencies
- slow-incident-resolution
- missing-rollback-strategy
- deadlock-conditions
- stack-overflow-errors
- race-conditions
- dma-coherency-issues
- incorrect-max-connection-pool-size
- lock-contention
- misconfigured-connection-pools
layout: solution
lang: de
en_slug: stress-testing
related_solutions:
- slug: load-testing
  similarity: 0.9
- slug: chaos-engineering
  similarity: 0.85
- slug: rate-limiting
  similarity: 0.75
- slug: integration-tests
  similarity: 0.75
- slug: cross-version-testing
  similarity: 0.75
- slug: resilience
  similarity: 0.75
---

## Description

Stresstests treiben ein System absichtlich über seine erwartete Spitzenlast hinaus — schrittweise Erhöhung des Traffics, Erschöpfung von Ressourcen wie Datenbankverbindungen oder Speicher, oder Einschleusen von Fehlern wie Prozessabbrüchen und Netzwerkpartitionen —, bis es degradiert oder bricht, um seine tatsächliche Kapazitätsobergrenze und Fehlermodi zu entdecken, statt anzunehmen, dass sie verstanden sind. Dies unterscheidet sich von gewöhnlichem Lasttesten in der Absicht: Das Ziel ist nicht zu bestätigen, dass das System erwarteten Traffic handhabt, sondern absichtlich herauszufinden, wo und wie es aufhört, Traffic zu handhaben, was Information ist, die nur durch tatsächliches Verursachen des Fehlschlags unter kontrollierten Bedingungen erhalten werden kann. Legacy-Systeme neigen besonders dazu, auf Weisen zu versagen, die niemand vorhersah, weil ihre ursprünglichen Kapazitätsannahmen vor langer Zeit gegen Traffic-Muster festgelegt wurden, die sich seitdem geändert haben, und die beteiligten Komponenten wurden oft nie mit elegantem Degradieren im Sinn gestaltet — eine Verbindungspool-Erschöpfung könnte beispielsweise einen unbehandelten Absturz statt eine kontrollierte Backpressure-Antwort auslösen. Das Durchführen von Stresstests bringt diese Fehlermodi zutage — ein Warteschlangen-Überlaufmechanismus, der still Nachrichten verwirft statt Backpressure anzuwenden, ein Absturz statt einer degradierten Antwort —, während das System unter Beobachtung in einer kontrollierten Umgebung ist, statt während eines echten Produktionsvorfalls, wenn dieselbe Entdeckung weit kostspieliger und weit weniger ruhig ist. Die Ergebnisse informieren direkt, wo Circuit Breaker, Auto-Scaling-Regeln und Alarmschwellen gesetzt werden sollten, aber die Praxis erfordert eine isolierte Umgebung, um die Beschädigung echter Daten oder Zustände zu vermeiden, und sie verbraucht erhebliche Infrastrukturressourcen, um bedeutsam ausgeführt zu werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Gestalten Sie Stresstests, die das System über die erwartete Spitzenlast hinaus treiben, um Bruchpunkte und Fehlermodi zu finden
- Erhöhen Sie die Last schrittweise, bis das System degradiert oder versagt, und protokollieren Sie Metriken auf jeder Ebene, um ein Kapazitätsprofil aufzubauen
- Testen Sie Fehler- und Wiederherstellungsverhalten: Was passiert, wenn der Datenbank die Verbindungen ausgehen, der Speicher erschöpft ist oder Festplatten sich füllen
- Beziehen Sie Chaos-Engineering-Elemente ein, wie das Beenden von Prozessen, das Einführen von Netzwerkpartitionen oder das Degradieren von Abhängigkeiten
- Führen Sie Stresstests gegen eine produktionsähnliche Umgebung mit repräsentativen Datenvolumina durch
- Dokumentieren Sie beobachtete Fehlermodi und ihre Symptome, um Playbooks zur Vorfallreaktion zu verbessern
- Nutzen Sie Stresstest-Ergebnisse, um Alarmschwellen zu etablieren und zu validieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Offenbart, wie das System versagt, und ermöglicht proaktive Härtung vor Produktionsvorfällen
- Identifiziert die tatsächliche Kapazitätsobergrenze, nicht nur den komfortablen Betriebsbereich
- Validiert, dass elegantes Degradieren und Circuit Breaker unter extremen Bedingungen funktionieren
- Verbessert das Vertrauen des Teams beim Umgang mit Produktionsnotfällen

**Kosten und Risiken:**
- Stresstests können Datenbeschädigung oder Zustandsinkonsistenzen in der Testumgebung verursachen
- Erfordert isolierte Umgebungen, um Auswirkungen auf andere Systeme zu verhindern
- Legacy-Systeme könnten während Stresstests auf destruktive Weisen versagen, was sorgfältige Vorbereitung erfordert
- Ergebnisse könnten für Stakeholder alarmierend sein, wenn sie nicht mit Kontext kommuniziert werden
- Die Durchführung von Stresstests erfordert erhebliche Infrastrukturressourcen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Zahlungsverarbeitungssystem hatte im letzten Jahr zwei größere Ausfälle während unerwarteter Traffic-Spitzen erlebt, aber das Team hatte kein Verständnis der tatsächlichen Grenzen des Systems. Sie führten eine Reihe von Stresstests durch, die das Transaktionsvolumen schrittweise von normalen Werten auf das 5-Fache der Spitzenlast erhöhten. Beim 2,5-Fachen der Spitzenlast entdeckten sie, dass der plattenbasierte Überlaufmechanismus der Legacy-Message-Queue einen Fehler hatte, der Nachrichtenverlust statt Backpressure verursachte. Beim 4-Fachen der Spitzenlast löste die Verbindungspool-Erschöpfung der Datenbank eine unbehandelte Exception aus, die den Anwendungsserver zum Absturz brachte, statt elegant zu degradieren. Beide Probleme wurden behoben, und die Stresstest-Ergebnisse informierten die Bereitstellung von Auto-Scaling-Regeln, die aktiviert wurden, bevor das System seine Bruchgrenze erreichte.
