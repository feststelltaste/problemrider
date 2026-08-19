---
title: Feature Detection
description: Abfrage von Systemfähigkeiten zur Laufzeit statt Verlass auf Versionsnummern.
category:
- Code
- Architecture
problems:
- technology-lock-in
- vendor-lock-in
- deployment-environment-inconsistencies
- inconsistent-behavior
- brittle-codebase
- hidden-dependencies
- dependency-version-conflicts
layout: solution
lang: de
en_slug: feature-detection
related_solutions:
- slug: cross-version-testing
  similarity: 0.7
- slug: compatibility-testing
  similarity: 0.7
- slug: documentation-of-compatibility-requirements
  similarity: 0.7
- slug: feature-toggles
  similarity: 0.7
- slug: forward-compatibility
  similarity: 0.7
- slug: compatibility-as-error
  similarity: 0.65
---

## Description

Feature Detection fragt eine Laufzeitumgebung ab, ob eine bestimmte Fähigkeit tatsächlich vorhanden ist — indem direkt auf eine konkrete API oder ein konkretes Verhalten geprüft wird — statt anhand einer Versionsnummer oder eines Bezeichners zu verzweigen und anzunehmen, was diese Version über verfügbare Funktionalität impliziert. Legacy-Codebasen, die anhand von Versionszeichenfolgen verzweigen, wie Browser-User-Agent-Sniffing oder OS-Versionsprüfungen, sind auf eine bestimmte Weise brüchig: Die angenommene Korrelation zwischen einer Versionsnummer und einer Fähigkeit bricht in dem Moment, in dem eine neue Version ändert, was sie unterstützt, oder ein zuvor zuverlässiger Bezeichner gefälscht oder abgekündigt wird, und jeder solche Bruch erfordert eine weitere Runde manueller Aktualisierungen der Versionsabgleichslogik. Diese Prüfungen durch direkte Fähigkeitsabfragen zu ersetzen, gekapselt hinter einer Abstraktionsschicht mit einem sanften Fallback für jedes erkannte Fehlen, beseitigt diesen Wartungsaufwand vollständig und lässt denselben Code korrekt über eine breitere und weniger vorhersagbare Bandbreite von Umgebungen laufen, sanft degradierend statt hart zu versagen, wo eine Fähigkeit fehlt. Die Kosten sind ein geringer Laufzeit-Overhead für die Abfragen selbst, weniger genutzte Fallback-Codepfade, die eigene latente Fehler verbergen können, und zusätzliche Verzweigungskomplexität durch die parallele Pflege mehrerer Ausführungspfade.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie plattformspezifische Bedingungen in der Codebasis, die Compile-Zeit-Flags oder Versionsprüfungen verwenden
- Ersetzen Sie versionsbasierte Verzweigung durch Laufzeit-Fähigkeitsabfragen, die prüfen, ob ein Feature oder eine API tatsächlich verfügbar ist
- Implementieren Sie sanfte Fallbacks für jede erkannte Fähigkeit, damit die Anwendung auf weniger leistungsfähigen Plattformen sanft degradiert
- Erstellen Sie eine Abstraktionsschicht, die die Feature-Detection-Logik kapselt und den Rest der Codebasis plattformunabhängig hält
- Fügen Sie Logging hinzu, wenn Fallbacks ausgelöst werden, damit das Team verfolgen kann, welchen Umgebungen erwartete Fähigkeiten fehlen
- Schreiben Sie Tests, die sowohl das Vorhandensein als auch das Fehlen von Plattformfunktionen simulieren, um das Fallback-Verhalten zu verifizieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt brüchige Versionsprüfungen, die brechen, wenn sich Plattformen weiterentwickeln oder auseinanderdriften
- Erlaubt der Anwendung, auf einer breiteren Bandbreite von Umgebungen zu laufen, ohne Codeänderungen
- Bietet sanfte Degradation statt harter Fehlschläge auf nicht unterstützten Plattformen
- Macht das System widerstandsfähiger gegenüber unerwarteten Umgebungsunterschieden

**Kosten und Risiken:**
- Laufzeiterkennung fügt Overhead im Vergleich zu Compile-Zeit-Entscheidungen hinzu, meist jedoch vernachlässigbar
- Fallback-Codepfade erhalten weniger Testabdeckung und können subtile Fehler verbergen
- Erhöhte Codekomplexität durch die Pflege mehrerer Ausführungspfade
- Manche Features können zur Laufzeit nicht sinnvoll abgefragt werden und erfordern weiterhin bedingte Kompilierung

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Webanwendung verließ sich auf Browser-User-Agent-Zeichenfolgen, um zu entscheiden, welche JavaScript-APIs verwendet werden sollten, was zu häufigen Brüchen bei neuen Browserversionen führte. Das Team ersetzte das User-Agent-Sniffing durch Modernizr-artige Feature Detection, die zur Laufzeit Fähigkeiten wie WebSocket-Unterstützung und CSS Grid abfragte. Wenn ein Feature fehlte, wich die Anwendung auf Polyfills oder einfachere Alternativen aus. Dies beseitigte den ständigen Wartungsaufwand der Aktualisierung von Browserversionslisten und reduzierte browserübergreifende Defektmeldungen um etwa 60 %.
