---
title: Fehlereindämmung
description: Begrenzung der Auswirkung von Fehlern auf einen kleinen Teil des Systems.
category:
- Architecture
problems:
- cascade-failures
- single-points-of-failure
- ripple-effect-of-changes
- monolithic-architecture-constraints
- tight-coupling-issues
- unpredictable-system-behavior
- system-outages
layout: solution
lang: de
en_slug: fault-containment
related_solutions:
- slug: isolation-of-faulty-components
  similarity: 0.8
- slug: bulkhead
  similarity: 0.8
- slug: resilience
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.75
- slug: containerization
  similarity: 0.7
- slug: chaos-engineering
  similarity: 0.7
---

## Description

Fehlereindämmung begrenzt den Explosionsradius eines Ausfalls auf die spezifische Komponente, in der er entstanden ist, mittels Techniken wie Bulkhead-Isolation von Ressourcen, separaten Thread- oder Verbindungspools pro Funktionsbereich, Circuit Breakern an Integrationsgrenzen und Isolation auf Prozess- oder Container-Ebene, sodass ein Fehler in einem Teil des Systems keine Ressourcen erschöpfen oder Ausfälle in unverbundene Teile propagieren kann. Dies adressiert ein prägendes Merkmal vieler Legacy-Monolithen: Komponenten, die nie mit Fehlerisolation im Blick entworfen wurden, teilen sich am Ende einen einzigen Prozess, Speicherbereich und Ressourcenpool, sodass ein Fehler in einer Randfunktion — etwa ein Reporting-Modul, dem der Speicher ausgeht — einen völlig unverbundenen kritischen Pfad lahmlegen kann, der zufällig denselben Server teilt. Die Einführung von Isolationsgrenzen um die risikoreichsten Komponenten, mit Timeouts bei jedem komponentenübergreifenden Aufruf, damit eine langsame Abhängigkeit nicht still alles nachgelagerte blockieren kann, verwandelt einen Alles-oder-Nichts-Ausfallmodus in einen eingedämmten, wiederherstellbaren, und schafft natürliche Nahtstellen, die später auch inkrementelle Modernisierung unterstützen. Die Kosten sind, dass das nachträgliche Einbauen von Isolation in einen Monolithen selbst beträchtliche Refactoring-Arbeit ist, Ressourcen über die neu getrennten Fehlerdomänen hinweg dupliziert und speziell dafür gebautes Monitoring erfordert, um Fehler zu erkennen, die konstruktionsbedingt nicht mehr als vollständiger Ausfall in Erscheinung treten.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Fehlerdomänen im Legacy-System, indem Sie analysieren, welche Komponenten sich Ressourcen, Threads oder Speicherbereiche teilen
- Führen Sie Bulkhead-Muster ein, um kritische Subsysteme zu isolieren, sodass ein Ausfall in einem nicht Ressourcen verbraucht, die andere benötigen
- Nutzen Sie separate Thread-Pools, Verbindungspools oder Prozessgrenzen für unabhängige Funktionsbereiche
- Wenden Sie Circuit Breaker an Integrationsgrenzen an, um die Ausfallpropagation zwischen Diensten zu stoppen
- Stellen Sie kritische Komponenten in isolierten Containern oder virtuellen Maschinen bereit, um Eindämmung auf Prozessebene durchzusetzen
- Fügen Sie Timeout-Richtlinien für alle komponentenübergreifenden Aufrufe hinzu, um zu verhindern, dass eine langsame Abhängigkeit das gesamte System blockiert
- Überprüfen Sie den Fehlerbehandlungscode, um sicherzustellen, dass Exceptions lokal abgefangen und behandelt statt ungeprüft weitergereicht werden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Begrenzt den Explosionsradius, sodass ein Fehler in einem Bereich nicht das gesamte System lahmlegt
- Macht das System unter partiellen Ausfallbedingungen vorhersagbarer
- Ermöglicht die unabhängige Wiederherstellung ausgefallener Komponenten
- Unterstützt inkrementelle Modernisierung durch die Schaffung natürlicher Grenzen

**Kosten und Risiken:**
- Das Einführen von Isolationsgrenzen in einen Monolithen erfordert erheblichen Refactoring-Aufwand
- Ressourcenduplikation über Fehlerdomänen hinweg erhöht den Gesamtressourcenverbrauch
- Übermäßige Isolation kann legitime übergreifende Operationen komplexer machen
- Teams brauchen Monitoring, um eingedämmte Fehler zu erkennen und darauf zu reagieren, die Nutzer möglicherweise nicht bemerken

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Gesundheitsplattform erlebte vollständige Ausfälle, wann immer ihr PDF-Berichtsgenerierungsmodul der Speicher ausging, weil es sich denselben Anwendungsserverprozess mit der Patientenakten-API teilte. Durch das Verschieben der Berichtsgenerierung in einen separaten Prozess mit eigenen Speichergrenzen und einem Circuit Breaker am Integrationspunkt dämmte das Team speicherbezogene Fehler auf das Reporting-Subsystem ein. Der Zugriff auf Patientenakten blieb selbst während Berichtsgenerierungsausfällen verfügbar, was den Schweregrad der Vorfälle von kritisch auf gering senkte.
