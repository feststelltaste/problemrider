---
title: API-Versionierungsstrategie
description: Wahl eines konkreten Mechanismus zur Identifikation und Weiterleitung
  zwischen API-Versionen.
category:
- Architecture
problems:
- api-versioning-conflicts
- legacy-api-versioning-nightmare
- breaking-changes
- poor-interfaces-between-applications
- integration-difficulties
- maintenance-overhead
- rapid-system-changes
- abi-compatibility-issues
layout: solution
lang: de
en_slug: api-versioning-strategy
related_solutions:
- slug: backward-compatible-apis
  similarity: 0.75
- slug: api-deprecation-policy
  similarity: 0.75
- slug: versioning-scheme
  similarity: 0.75
- slug: backward-compatibility
  similarity: 0.7
- slug: version-control
  similarity: 0.7
- slug: content-negotiation
  similarity: 0.7
---

## Description

Eine API-Versionierungsstrategie ist eine explizite, organisationsweite Entscheidung darüber, wie API-Versionen identifiziert und weitergeleitet werden — über URL-Pfad, Query-Parameter, benutzerdefinierten Header oder Content Negotiation — gepaart mit einer klaren, dokumentierten Definition dessen, was einen Breaking Change im Vergleich zu einer nicht brechenden Änderung ausmacht. Ohne eine solche Strategie neigen Legacy-Systeme dazu, Versionierungskonventionen ad hoc anzuhäufen, da verschiedene Teams oder sogar verschiedene Entwickler im selben Team ihren eigenen Ansatz erfinden, wann immer ein Breaking Change unvermeidlich wird, was die Codebasis mit mehreren inkompatiblen, übereinander geschichteten Versionierungsschemata zurücklässt. Die Wahl eines einzigen Mechanismus und dessen Dokumentation als verbindlicher Standard beseitigt diese Inkonsistenz und erlaubt es, Versions-Routing einmal in einer zentralisierten Schicht wie einem API-Gateway oder Reverse-Proxy zu implementieren, statt es in jedem Service unterschiedlich neu zu implementieren. Dies ist besonders wichtig während der Legacy-Modernisierung, wenn neue Implementierungen schrittweise neben alten ausgerollt werden und Konsumenten eine stabile, vorhersagbare Möglichkeit brauchen zu wissen, mit welchem Vertrag sie sprechen und wann sie migrieren müssen. Eine Versionierungsstrategie allein verhindert jedoch keine Versionswucherung; sie muss mit einer Deprecation-Richtlinie gepaart werden, die alte Versionen tatsächlich ausmustert, sonst wird die Anzahl gleichzeitig unterstützter Versionen weiter wachsen, unabhängig davon, wie konsistent sie beschriftet sind.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Bewerten Sie Versionierungsansätze (URL-Pfad, Query-Parameter, Header, Content Negotiation) gegen Ihre Konsumentenbasis und Infrastruktur
- Wählen Sie einen Mechanismus und dokumentieren Sie ihn als verbindlichen Standard für alle Teams
- Implementieren Sie Versions-Routing in einer zentralisierten Schicht (z. B. API-Gateway oder Reverse-Proxy), statt Logik über Services zu verstreuen
- Definieren Sie, was einen Breaking Change im Vergleich zu einer nicht brechenden Änderung ausmacht, und dokumentieren Sie die Regeln
- Bieten Sie versionsspezifische Dokumentation und Changelogs für jede unterstützte API-Version
- Kombinieren Sie die Versionierungsstrategie mit einer Deprecation-Richtlinie, um unbegrenzte Versionswucherung zu verhindern

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Gibt Konsumenten Stabilität, während dem Backend erlaubt wird, sich weiterzuentwickeln
- Macht Breaking Changes explizit und handhabbar statt zufällig
- Ermöglicht schrittweise Migration von Konsumenten zu neueren Versionen

**Kosten und Risiken:**
- Mehrere lebende Versionen erhöhen die Test- und Wartungslast
- Teams könnten Migrationen aufschieben, wodurch alte Versionen unbegrenzt am Leben bleiben
- Inkonsistente Übernahme über Teams hinweg untergräbt den Wert der Strategie
- Manche Versionierungsmechanismen (z. B. URL-Pfad) können zu Code-Duplizierung in Service-Implementierungen führen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein SaaS-Unternehmen mit über 200 API-Konsumenten hatte keine formale Versionierungsstrategie, was zu häufigen unangekündigten Breaking Changes führte, die Produktionsvorfälle für nachgelagerte Clients verursachten. Das Team übernahm URL-Pfad-Versionierung mit maximal drei gleichzeitig unterstützten Versionen und einem 9-monatigen Deprecation-Fenster. Innerhalb eines Jahres sanken von Konsumenten gemeldete Integrationsfehler um 70 %, und das Team verringerte die Anzahl der Legacy-API-Varianten von elf auf drei.
