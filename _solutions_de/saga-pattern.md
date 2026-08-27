---
title: Saga Pattern
description: Verwaltung verteilter Transaktionen durch Sequenzen lokaler
  Transaktionen mit kompensierenden Aktionen.
category:
- Architecture
problems:
- cascade-failures
- long-running-transactions
- tight-coupling-issues
- unpredictable-system-behavior
- microservice-communication-overhead
- data-migration-integrity-issues
- deadlock-conditions
layout: solution
lang: de
en_slug: saga-pattern
related_solutions:
- slug: transactions
  similarity: 0.75
- slug: retry
  similarity: 0.75
- slug: data-replication
  similarity: 0.7
- slug: distributed-processing
  similarity: 0.7
- slug: business-event-processing
  similarity: 0.7
- slug: event-driven-architecture
  similarity: 0.7
---

## Description

Das Saga Pattern verwaltet eine Geschäftstransaktion, die mehrere Dienste oder Datenbanken umspannt, indem sie in eine Sequenz lokaler Transaktionen zerlegt wird, von denen jede mit einer kompensierenden Aktion gepaart ist, die ihre Effekte rückgängig machen kann, falls ein späterer Schritt in der Sequenz fehlschlägt, und erreicht dadurch eventuelle Konsistenz, ohne sich auf ein verteiltes Two-Phase-Commit zu verlassen. Die Sequenz kann entweder durch Choreografie koordiniert werden, wo jeder Dienst auf vom vorherigen Schritt emittierte Ereignisse reagiert, oder durch Orchestrierung, wo ein zentraler Koordinator jeden Schritt explizit lenkt und den Gesamtzustand der Saga verfolgt. Dieses Muster wird speziell notwendig, wenn ein Legacy-System, das sich einst auf eine einzelne Datenbanktransaktion verließ, um Atomizität über mehrere Operationen hinweg zu garantieren, während der Modernisierung in separate Dienste zerlegt wird, da diese Zerlegung die ursprüngliche transaktionale Garantie bricht und einen expliziten Mechanismus erfordert, um äquivalente Konsistenz wiederherzustellen. Da jede lokale Transaktion den für sie am besten geeigneten Datenspeicher und die Isolationsstufe nutzen kann, beseitigt das Saga Pattern auch die enge Kopplung, die eine gemeinsame verteilte Transaktion sonst über die neu getrennten Dienste auferlegen würde, was oft eines der Hauptziele der Zerlegung überhaupt ist. Die Kosten des Musters sind eine echte Zunahme der Designkomplexität: Kompensierende Aktionen müssen für jeden Schritt entworfen werden, manche Operationen — wie eine gesendete E-Mail oder eine versandte physische Ware — können überhaupt nicht sauber kompensiert werden, und die resultierende vorübergehende Inkonsistenz zwischen Saga-Schritten, zusammen mit der Schwierigkeit, eine fehlgeschlagene Multi-Service-Saga zu debuggen, sind Preise, die gegen die Kopplung abgewogen werden müssen, die das Muster beseitigt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie verteilte Transaktionen im Legacy-System, die mehrere Dienste oder Datenbanken umspannen
- Zerlegen Sie jede verteilte Transaktion in eine Sequenz lokaler Transaktionen mit definierter Reihenfolge
- Entwerfen Sie kompensierende Aktionen für jeden Schritt, die ihre Effekte rückgängig machen können, falls ein nachfolgender Schritt fehlschlägt
- Wählen Sie zwischen Choreografie (ereignisgetrieben) und Orchestrierung (zentraler Koordinator) basierend auf der Systemkomplexität
- Implementieren Sie idempotente Operationen bei jedem Schritt, um Wiederholungen sicher zu handhaben
- Fügen Sie Überwachung und Alarmierung für Sagas hinzu, die über erwartete Dauern hinaus in Zwischenzuständen verbleiben
- Speichern Sie den Saga-Zustand persistent, um Prozessneustarts zu überleben und die Wiederherstellung laufender Sagas zu ermöglichen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt die Notwendigkeit für verteiltes Two-Phase-Commit, das in Legacy-Umgebungen fragil ist
- Ermöglicht Datenkonsistenz über Dienstgrenzen hinweg ohne enge Kopplung
- Jede lokale Transaktion kann den am besten geeigneten Datenspeicher und die Isolationsstufe nutzen
- Fehlgeschlagene Transaktionen werden automatisch kompensiert, statt in inkonsistenten Zuständen belassen zu werden

**Kosten und Risiken:**
- Kompensierende Aktionen fügen erhebliche Design- und Implementierungskomplexität hinzu
- Vorübergehende Dateninkonsistenz ist zwischen Saga-Schritten sichtbar (eventuelle Konsistenz)
- Das Debuggen fehlgeschlagener Sagas über mehrere Dienste hinweg ist herausfordernd
- Manche Operationen sind schwer oder unmöglich zu kompensieren (gesendete E-Mails, versandte physische Waren)

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Buchungssystem eines Reisebüros nutzte eine einzelne Datenbanktransaktion, um Flüge, Hotels und Mietwagen gleichzeitig zu reservieren. Als das System in separate Dienste zerlegt wurde, brach die monolithische Transaktion. Das Team implementierte ein Saga Pattern, bei dem jeder Buchungsschritt eine lokale Transaktion mit einer kompensierenden Stornierungsaktion war. Wenn die Hotelreservierung erfolgreich war, aber der Mietwagen fehlschlug, stornierte die Saga automatisch die Hotelreservierung und benachrichtigte den Flugdienst, den Sitz freizugeben. Ein Orchestrator-Dienst verfolgte den Saga-Zustand und handhabte Wiederholungen für vorübergehende Fehler, was dieselbe Alles-oder-Nichts-Buchungssemantik ohne verteilte Transaktionen lieferte.
