---
title: Abstraktion
description: Entkopplung von Komponenten durch Verträge, sodass Implementierungen
  unabhängig voneinander variieren können.
category:
- Architecture
- Code
problems:
- high-coupling-low-cohesion
- tight-coupling-issues
- ripple-effect-of-changes
- monolithic-architecture-constraints
- technology-lock-in
- vendor-lock-in
- difficult-code-reuse
- stagnant-architecture
- poor-encapsulation
layout: solution
lang: de
en_slug: abstraction
related_solutions:
- slug: protocol-abstraction
  similarity: 0.85
- slug: abstraction-layers
  similarity: 0.8
- slug: database-abstraction
  similarity: 0.8
- slug: loose-coupling
  similarity: 0.8
- slug: bridges
  similarity: 0.75
- slug: facades
  similarity: 0.75
---

## Description

Abstraktion ist die allgemeine Praxis, explizite Schnittstellen oder Verträge an den Grenzen zwischen Komponenten zu definieren, sodass jede Seite nur vom vereinbarten Vertrag abhängt statt von den konkreten Implementierungsdetails der anderen Seite. Sobald ein solcher Vertrag besteht, kann sich jede Seite intern ändern — eine Datenstruktur austauschen, eine Bibliothek ersetzen, einen Algorithmus umschreiben —, solange sie weiterhin den Vertrag einhält, was das Tempo und Risiko der Änderung auf einer Seite von der anderen entkoppelt. Legacy-Systeme neigen dazu, über die Zeit den gegenteiligen Zustand anzuhäufen: Module greifen direkt in die Internas des jeweils anderen, Geschäftslogik instanziiert konkrete Anbieterklassen, und eine Änderung irgendwo pflanzt sich unvorhersehbar überall fort, weil nie eine stabile Grenze etabliert wurde. Die Einführung von Abstraktion an diesen Grenzen erfolgt üblicherweise inkrementell, oft als Teil einer Strangler-Fig-Migration, indem eine bestehende Komponente hinter einer neu definierten Schnittstelle eingehüllt wird, bevor ihre Internas berührt werden, was die starre interne Struktur eines Monolithen in eine Menge unabhängig ersetzbarer Teile verwandelt. Dies ist außerdem, was viele andere strukturelle Abhilfen überhaupt erst möglich macht: Dependency Injection, Mocking in Tests und Anbieterersatz hängen alle von einem vorherigen Abstraktionsschritt ab, an dem sie ansetzen können. Weil Verträge nur nützlich sind, wenn sie stabil bleiben, kann die verfrühte Einführung von Abstraktion — bevor klar ist, wo tatsächlich Variation benötigt wird — Komplexität ohne entsprechenden Nutzen hinzufügen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie eng gekoppelte Grenzen im Legacy-System und definieren Sie explizite Schnittstellen oder Verträge zwischen ihnen
- Führen Sie Schnittstellentypen oder abstrakte Basisklassen an Modulgrenzen ein, bevor Sie Implementierungen ändern
- Ersetzen Sie direkte Klasseninstanziierung durch Dependency Injection oder Factory-Muster
- Extrahieren Sie plattform- oder anbieterspezifischen Code hinter Abstraktionsschichten, sodass Alternativen ausgetauscht werden können
- Nutzen Sie den Strangler-Fig-Ansatz, um Legacy-Komponenten graduell mit sauberen Abstraktionen zu umhüllen
- Schreiben Sie Integrationstests gegen den Vertrag statt gegen die Implementierung, um Substituierbarkeit zu validieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht unabhängige Weiterentwicklung von Komponenten, was den Wirkungsradius von Änderungen verringert
- Macht es möglich, Legacy-Implementierungen inkrementell zu ersetzen, ohne Big-Bang-Neuschreibungen
- Verbessert die Testbarkeit, indem Mock- oder Stub-Implementierungen erlaubt werden
- Verringert Vendor Lock-in, indem Implementierungsdetails hinter stabilen Verträgen gehalten werden

**Kosten und Risiken:**
- Fügt Indirektion hinzu, die Debugging und Nachverfolgung in unvertrauten Codebasen erschweren kann
- Verfrühte Abstraktion kann unnötige Komplexität schaffen, wenn sich die Variationspunkte nie materialisieren
- Erfordert Teamdisziplin, um Verträge stabil und gut dokumentiert zu halten
- Performance-sensible Pfade können unter dem Overhead zusätzlicher Schichten leiden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Logistikunternehmen hatte sein Bestellverarbeitungssystem direkt an ein spezifisches Message-Queue-Produkt gekoppelt. Als der Anbieter die Preise erheblich erhöhte, wurde ein Wechsel auf sechs Monate Arbeit geschätzt. Durch die Einführung einer Messaging-Abstraktionsschicht über einen Zeitraum von drei Monaten konnte das Team den zugrunde liegenden Broker in zwei Wochen austauschen. Dieselbe Abstraktion erlaubte es ihnen später, während Integrationstests eine In-Memory-Implementierung laufen zu lassen, was die Ausführungszeit der Testsuite um 60 % verkürzte.
