---
title: Monitoring
description: Kontinuierliche Überwachung von Systemzuständen, Performance und Fehlern.
category:
- Operations
problems:
- monitoring-gaps
- slow-incident-resolution
- constant-firefighting
- system-outages
- gradual-performance-degradation
- unpredictable-system-behavior
- high-defect-rate-in-production
- poor-operational-concept
- cache-invalidation-problems
- database-connection-leaks
- deadlock-conditions
- index-fragmentation
- inefficient-database-indexing
- load-balancing-problems
- poor-caching-strategy
- synchronization-problems
- unused-indexes
- upstream-timeouts
- log-spam
- long-running-database-transactions
- race-conditions
- dma-coherency-issues
- excessive-logging
- lock-contention
- long-running-transactions
- rate-limiting-issues
- service-discovery-failures
layout: solution
lang: de
en_slug: monitoring
related_solutions:
- slug: continuous-performance-monitoring
  similarity: 0.85
- slug: monitoring-system-utilization
  similarity: 0.85
- slug: logging
  similarity: 0.8
- slug: performance-measurements
  similarity: 0.8
- slug: security-monitoring
  similarity: 0.8
- slug: monitoring-system-integrity
  similarity: 0.8
---

## Description

Monitoring ist die kontinuierliche Sammlung und Beobachtung der technischen und geschäftlichen Signale eines Systems — Metriken, Logs, Traces, Fehlerraten, Antwortzeiten und Ressourcennutzung —, sichtbar gemacht über Dashboards und Alarmierung, damit Probleme proaktiv erkannt und diagnostiziert werden können, statt erst entdeckt zu werden, wenn ein Nutzer sie meldet. Es umzusetzen bedeutet, Anwendungen zu instrumentieren, um Metriken auszusenden, Logs zentral über alle Komponenten hinweg zu aggregieren, verteiltes Tracing hinzuzufügen, damit eine einzelne Anfrage über Dienstgrenzen hinweg verfolgt werden kann, und Alarmschwellenwerte und -schweregrade so abzustimmen, dass das Signal die richtigen Personen mit der richtigen Dringlichkeit erreicht. Legacy-Systeme werden häufig mit Monitoring betrieben, das kaum mehr ist als die Bestätigung, dass ein Prozess noch läuft, was bedeutet, dass das Team keine Sichtbarkeit auf graduelle Verschlechterung, Ressourcenerschöpfung oder intermittierende Fehler hat, bis sie zu einem Vorfall eskalieren, der schwer genug ist, dass ihn jemand nachgelagert bemerkt und meldet. Echtes Monitoring über ein solches System zu etablieren ist oft der erste Schritt mit der höchsten Hebelwirkung in jedem Modernisierungsvorhaben, weil er Jahre undurchsichtigen, undokumentierten Laufzeitverhaltens in beobachtbare Daten umwandelt — Speicherlecks, langsame Abfragen, die mit dem Datenwachstum verkommen, Race Conditions —, die dann mit Evidenz statt Vermutung diagnostiziert und behoben werden können. Das Risiko, dies schlecht zu tun, ist, dass Monitoring, einmal vorhanden, ebenso leicht falsches Vertrauen oder Alarmmüdigkeit erzeugen kann wie es Einsicht erzeugen kann, sodass die Instrumentierung mit disziplinierter Schwellenwertüberprüfung gepaart werden muss, um das Signal-Rausch-Verhältnis handhabbar zu halten, während sich das Legacy-System und seine Fehlermodi weiterentwickeln.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Instrumentieren Sie Legacy-Anwendungen mit Metrikerfassung für wichtige geschäftliche und technische Indikatoren
- Setzen Sie zentralisierte Log-Aggregation ein, um Logs aus allen Legacy-Systemkomponenten zu konsolidieren
- Erstellen Sie Dashboards, die Systemgesundheit, Fehlerraten, Antwortzeiten und Ressourcennutzung anzeigen
- Richten Sie Alarmierungsregeln mit angemessenen Schweregraden und Benachrichtigungskanälen ein
- Überwachen Sie sowohl Infrastrukturmetriken (CPU, Speicher, Festplatte) als auch Anwendungsmetriken (Anfrageraten, Fehlerraten, Latenz)
- Fügen Sie verteiltes Tracing hinzu, um Anfragen über Legacy-Systemgrenzen hinweg zu verfolgen
- Überprüfen und stimmen Sie Alarmschwellenwerte regelmäßig ab, um Rauschen zu verringern und Alarmmüdigkeit zu verhindern
- Beziehen Sie Geschäftsmetriken (Auftragszahlen, Transaktionswerte) neben technischem Monitoring ein

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht proaktive Erkennung von Problemen, bevor sie zu nutzersichtbaren Vorfällen werden
- Liefert Daten für Ursachenanalyse und Trendidentifikation
- Verringert die mittlere Zeit bis zur Erkennung und Lösung von Produktionsproblemen
- Unterstützt Kapazitätsplanung mit historischen Nutzungsdaten
- Schafft Sichtbarkeit auf Legacy-Systemverhalten, das jahrelang undurchsichtig gewesen sein könnte

**Kosten und Risiken:**
- Monitoring-Infrastruktur erfordert eigene Pflege und Kapazitätsplanung
- Übermäßiges Monitoring kann Alarmmüdigkeit erzeugen, was Teams dazu bringt, Warnungen zu ignorieren
- Die Instrumentierung von Legacy-Anwendungen kann Codeänderungen oder Wrapper-Skripte erfordern
- Speicherkosten für Metriken und Logs können über die Zeit erheblich wachsen
- Schlecht konfiguriertes Monitoring liefert falsches Vertrauen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Logistikunternehmen betrieb ein Legacy-Lagerverwaltungssystem ohne Monitoring über die Prüfung hinaus, ob der Prozess lief. Probleme wurden erst entdeckt, wenn Lagerarbeiter Fehler oder fehlende Daten meldeten. Nach dem Einsatz von Monitoring, das Auftragsverarbeitungsraten, Datenbankabfragelatenzen und Fehler-Logs verfolgte, gewann das Team Sichtbarkeit auf ein langsames Speicherleck, das wöchentliche Neustarts verursacht hatte, und eine Datenbankabfrage, die verkam, während der Bestand wuchs. Mit diesen Daten behoben sie beide Probleme proaktiv und etablierten Alarmierung, die künftige Probleme Minuten statt Stunden nach ihrem Auftreten fing.
