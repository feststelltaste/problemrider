---
title: Lasttests
description: Bewertung von Systemperformance und -stabilität unter hoher Last.
category:
- Testing
- Performance
problems:
- capacity-mismatch
- slow-application-performance
- gradual-performance-degradation
- scaling-inefficiencies
- system-outages
- deployment-risk
- unpredictable-system-behavior
- database-connection-leaks
- incorrect-max-connection-pool-size
- inefficient-database-indexing
- load-balancing-problems
- misconfigured-connection-pools
- algorithmic-complexity-problems
- garbage-collection-pressure
- inefficient-code
- insufficient-worker-capacity
- memory-fragmentation
- atomic-operation-overhead
- data-structure-cache-inefficiency
- false-sharing
- improper-event-listener-management
- incorrect-index-type
- interrupt-overhead
- memory-barrier-inefficiency
- poor-caching-strategy
- rate-limiting-issues
- resource-allocation-failures
- serialization-deserialization-bottlenecks
- unreleased-resources
- unused-indexes
layout: solution
lang: de
en_slug: load-testing
related_solutions:
- slug: stress-testing
  similarity: 0.9
- slug: chaos-engineering
  similarity: 0.8
- slug: continuous-performance-monitoring
  similarity: 0.75
- slug: performance-modeling
  similarity: 0.75
- slug: performance-measurements
  similarity: 0.75
- slug: compatibility-testing
  similarity: 0.75
---

## Description

Lasttests unterziehen ein System simuliertem Verkehr — Anfragevolumina, gleichzeitige Nutzer und Datengrößen, die reale Produktionsbedingungen annähern oder übersteigen sollen —, um zu beobachten, wie es sich verhält und wo es bricht, bevor diese Bedingungen ungeplant in der Produktion auftreten. Werkzeuge wie JMeter, Gatling oder k6 erzeugen die synthetische Last gegen realistische Szenarien und produktionsähnliche Datenvolumina, während das Team Antwortzeiten, Fehlerraten und Ressourcensättigung beobachtet, um eine Performance-Basislinie zu etablieren und Regressionen in nachfolgenden Läufen zu erkennen, einschließlich verlängerter Soak-Tests, die darauf ausgelegt sind, langsame Lecks aufzudecken, die sich erst nach anhaltendem Betrieb manifestieren. Legacy-Systeme werden häufig deployt und jahrelang laufen gelassen, ohne dass ihre tatsächlichen Kapazitätsgrenzen je gemessen wurden, weil der ursprüngliche Lasttest (falls einer durchgeführt wurde) Verkehrsmuster und Datenvolumina von einem viel früheren Punkt in der Lebensdauer des Systems widerspiegelte, was das Team die echten Grenzen erst entdecken lässt, wenn eine saisonale Spitze oder unerwartete Welle das System über einen Schwellenwert drückt, von dem niemand wusste, dass er existiert. Lasttests bewusst gegen ein solches System durchzuführen, vor einem bekannten Hochnachfrage-Ereignis, verwandelt einen unbekannten und oft katastrophalen Ausfallmodus — Erschöpfung des Verbindungspools, Tabellensperrung unter Nebenläufigkeit, eine Reporting-Abfrage, die erst im großen Maßstab langsam wird — in einen bekannten, behebbaren Defekt, entdeckt unter kontrollierten Bedingungen. Weil Legacy-Systemen oft eine Testumgebung fehlt, die die Produktionstopologie und den Datenmaßstab getreu widerspiegelt, ist das größte praktische Hindernis für ihre Lasttests meist nicht das Tooling, sondern das Zusammenstellen einer Umgebung und eines Datensatzes, die realistisch genug sind, dass den Ergebnissen vertraut werden kann.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie realistische Lastprofile basierend auf tatsächlichen Produktionsverkehrsmustern und antizipiertem Wachstum
- Erstellen Sie Lasttest-Szenarien, die kritische Legacy-System-Pfade einschließlich Datenbankabfragen und Integrationen ausüben
- Nutzen Sie Lasttest-Werkzeuge (JMeter, Gatling, k6), um gleichzeitige Nutzer und anhaltenden Durchsatz zu simulieren
- Etablieren Sie Performance-Basislinien und setzen Sie Regressionsschwellenwerte, die CI/CD-Pipelines fehlschlagen lassen, wenn überschritten
- Testen Sie mit produktionsähnlichen Datenvolumina, da Legacy-Systeme mit Datenwachstum oft verkommen
- Beziehen Sie Soak-Tests (verlängerte Dauer) ein, um Speicherlecks und Ressourcenerschöpfung in Legacy-Code zu erkennen
- Führen Sie Lasttests in Umgebungen durch, die der Produktionstopologie so nah wie möglich entsprechen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Deckt Performance-Engpässe auf, bevor sie Produktionsnutzer betreffen
- Liefert datengestützte Kapazitätsplanungs-Inputs für Infrastrukturentscheidungen
- Validiert, dass Änderungen an Legacy-Systemen keine Performance-Regressionen einführen
- Baut Vertrauen für Produktions-Deployments und Skalierungsentscheidungen auf

**Kosten und Risiken:**
- Erfordert dedizierte Testumgebungen mit produktionsähnlichen Daten und Infrastruktur
- Lasttest-Pflege wird zu laufenden Kosten, während sich das System weiterentwickelt
- Tests könnten Produktionsbedingungen nicht perfekt replizieren, was falsches Vertrauen erzeugt
- Das Ausführen von Lasttests gegen gemeinsam genutzte Umgebungen kann andere Teams stören
- Legacy-Datenbankzustand nach Lasttests erfordert Bereinigung

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Regierungsdienste-Portal erlebte jährliche Ausfälle während Steuererklärungsfristen. Das Legacy-System war nie lastgetestet worden, und das Team hatte keine Daten über seine tatsächlichen Kapazitätsgrenzen. Durch die Implementierung von Lasttests, die den Spitzenverkehr bei Steuererklärungen simulierten, entdeckten sie, dass der Datenbankverbindungspool bei 40 % der erwarteten Spitzenlast erschöpft war und dass eine bestimmte Reporting-Abfrage Tabellensperren unter hoher Nebenläufigkeit verursachte. Die Behebung dieser Probleme vor der nächsten Frist führte zur ersten Erklärungssaison ohne Ausfallzeit seit fünf Jahren.
