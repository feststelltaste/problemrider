---
title: Adapter
description: Übersetzung zwischen inkompatiblen Schnittstellen durch eine vermittelnde
  Schicht.
category:
- Architecture
- Code
problems:
- poor-interfaces-between-applications
- integration-difficulties
- architectural-mismatch
- legacy-api-versioning-nightmare
- technology-stack-fragmentation
- breaking-changes
- vendor-dependency
- dependency-on-supplier
layout: solution
lang: de
en_slug: adapter
related_solutions:
- slug: abstraction-layers
  similarity: 0.8
- slug: facades
  similarity: 0.8
- slug: protocol-abstraction
  similarity: 0.8
- slug: api-gateway
  similarity: 0.8
- slug: dependency-injection
  similarity: 0.75
- slug: mediator
  similarity: 0.75
---

## Description

Das Adapter-Muster führt eine schmale Übersetzungsklasse oder ein -modul ein, das die vom konsumierenden Code erwartete Schnittstelle implementiert, während es intern an eine Komponente delegiert, deren bestehende Schnittstelle nicht passt — es konvertiert Aufrufe, Parameter und Rückgabewerte zwischen den beiden Formen, ohne eigene Geschäftslogik hinzuzufügen. Es ist eines der direktesten Werkzeuge zur Integration einer Legacy-Komponente in eine neuere Architektur, weil es der Legacy-Seite erlaubt, vollständig unberührt zu bleiben, während dem Rest des Systems eine saubere, zweckgebaute Schnittstelle zum Abhängen gegeben wird. Dies ist besonders wertvoll in der Legacy-Modernisierung, wenn die ursprüngliche Schnittstelle einer Komponente für eine Technologie oder ein Protokoll designt wurde, das nicht mehr passt, wie der Rest des Systems kommuniziert — ein Mainframe, der Copybook-Datensätze fester Breite nutzt, ein SOAP-Service in einer REST-orientierten Landschaft, oder eine Drittanbieterbibliothek, deren API-Design nicht zu den eigenen Konventionen der Anwendung passt. Durch das Umhüllen einer solchen Abhängigkeit hinter einem Adapter, der die Schnittstelle offenlegt, die die Anwendung tatsächlich möchte, werden Breaking Changes und anbieterspezifische Eigenheiten an einem einzigen, gut definierten Übersetzungspunkt absorbiert, statt in der gesamten Codebasis zu lecken. Adapter ermöglichen außerdem parallele Entwicklung, da ein Team sofort gegen die Zielschnittstelle bauen kann, während der Adapter unabhängig entwickelt wird, um die Lücke zur Legacy-Seite zu überbrücken. Weil ein Adapter nur Struktur übersetzt, muss er einfach und leicht isoliert testbar gehalten werden; das Einschleichen von Geschäftsregeln in die Übersetzungsschicht oder das Anhäufen zu vieler undisziplinierter Adapter schafft dieselben Kopplungsprobleme neu, die das Muster lösen sollte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie Integrationspunkte, an denen Legacy-Schnittstellen nicht dem entsprechen, was konsumierender Code erwartet
- Erstellen Sie Adapterklassen oder -module, die die Zielschnittstelle implementieren und an die Legacy-Komponente delegieren
- Halten Sie den Adapter schlank und führen Sie nur strukturelle Übersetzung ohne Hinzufügung von Geschäftslogik durch
- Nutzen Sie Adapter zur Umhüllung von Drittanbieterbibliotheken, sodass Ihre Codebasis von Ihrer eigenen Schnittstelle abhängt, nicht der des Anbieters
- Führen Sie Adapter inkrementell zuerst an den schmerzhaftesten Integrationsgrenzen ein
- Schreiben Sie Tests, die verifizieren, dass der Adapter korrekt zwischen beiden Schnittstellenverträgen übersetzt

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Erlaubt Legacy-Komponenten, an modernen Architekturen teilzunehmen, ohne sie neu zu schreiben
- Isoliert Breaking Changes von externen Systemen an einem einzigen Übersetzungspunkt
- Ermöglicht parallele Entwicklung: Teams können gegen die Zielschnittstelle programmieren, während der Adapter die Lücke überbrückt

**Kosten und Risiken:**
- Jeder Adapter fügt eine Wartungsfläche hinzu, die mit beiden Seiten synchron gehalten werden muss
- Adapter können tiefere Designprobleme maskieren und notwendiges Refactoring verzögern
- Schlecht designte Adapter können subtile Datenverluste oder semantische Fehlanpassungen einführen
- Eine Verbreitung von Adaptern kann eigene Komplexität schaffen, wenn sie nicht kontrolliert wird

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Finanzdienstleistungsunternehmen musste ein 15 Jahre altes COBOL-basiertes Kontoverwaltungssystem mit einem neuen REST-basierten Kundenportal integrieren. Statt das COBOL-System neu zu schreiben, baute das Team eine Reihe von Adaptern, die REST-Aufrufe in das COBOL-Copybook-Format übersetzten und Antworten zurück auf JSON abbildeten. Dies erlaubte dem neuen Portal, planmäßig zu launchen, während das Legacy-System unverändert weiter operierte, und es gab dem Team eine klare Naht für zukünftigen inkrementellen Ersatz.
