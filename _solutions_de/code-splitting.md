---
title: Code Splitting
description: Aufteilung des Anwendungscodes in kleinere Chunks.
category:
- Performance
- Code
problems:
- slow-application-performance
- high-client-side-resource-consumption
- inefficient-frontend-code
- gradual-performance-degradation
- feature-bloat
- high-resource-utilization-on-client
layout: solution
lang: de
en_slug: code-splitting
related_solutions:
- slug: tree-shaking
  similarity: 0.8
- slug: lazy-loading
  similarity: 0.8
- slug: image-and-asset-optimization
  similarity: 0.8
- slug: lazy-evaluation
  similarity: 0.75
- slug: predictive-loading
  similarity: 0.7
- slug: performance-optimization
  similarity: 0.7
---

## Description

Code Splitting ist eine Build-Zeit-Technik, die die kompilierte Ausgabe einer Anwendung in mehrere kleinere Chunks aufteilt, bei Bedarf geladen statt die gesamte Anwendung als ein monolithisches Bundle bei jedem Seitenbesuch auszuliefern. Routenbasiertes Splitting stellt sicher, dass eine gegebene Seite nur den Code lädt, den diese Seite tatsächlich braucht, während dynamische Imports das Laden von Features verzögern, die beim ersten Rendering nicht benötigt werden — Admin-Panels, selten genutzte Werkzeuge, modale Dialoge — bis zu dem Moment, in dem ein Nutzer sie tatsächlich erreicht. Dies ist besonders wichtig in Legacy-Single-Page-Anwendungen, die häufig über Jahre Features angehäuft haben, ohne Aufmerksamkeit für die Bundle-Größe, was dazu führt, dass jeder Nutzer Megabytes an JavaScript für Funktionalität herunterlädt, die nur ein kleiner Bruchteil von ihnen je berühren wird — unnötige Kosten, die schlimmer werden, je länger die Legacy-Anwendung erweitert wurde. Das Aufteilen von Vendor-Bibliotheken in ihren eigenen Chunk, getrennt vom Anwendungscode, verbessert außerdem das Caching, da ein selten geänderter Vendor-Bundle nicht bei jedem Anwendungscode-Update erneut heruntergeladen werden muss. Weil es auf Build-Konfigurationsebene operiert, kann Code Splitting typischerweise eingeführt werden, ohne die tatsächliche Logik der Anwendung neu zu designen, was es zu einer vergleichsweise risikoarmen Performance-Intervention gegenüber tieferen architektonischen Änderungen macht. Sein Hauptrisiko ist Über-Splitting: Das Aufteilen des Bundles in zu viele kleine Chunks tauscht ein Performance-Problem (ein großer initialer Download) gegen ein anderes (übermäßig viele Netzwerkanfragen), sodass die Split-Grenzen gegen tatsächliche Nutzungsmuster abgestimmt werden müssen, statt mechanisch überall angewendet zu werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Analysieren Sie das Anwendungsbundle, um große Module und Abhängigkeiten zu identifizieren, die am meisten zur initialen Ladegröße beitragen
- Implementieren Sie routenbasiertes Splitting, sodass jede Seite nur den benötigten Code lädt
- Nutzen Sie dynamische Imports für Features, die beim ersten Rendering nicht benötigt werden: Modals, Admin-Panels, selten genutzte Werkzeuge
- Teilen Sie Vendor-Bibliotheken in einen separaten Chunk auf, der unabhängig vom Anwendungscode gecacht werden kann
- Konfigurieren Sie das Build-Werkzeug (Webpack, Vite, esbuild), um angemessene Chunk-Größenlimits und Benennungsstrategien zu setzen
- Implementieren Sie Prefetching für Code-Chunks, die der Nutzer basierend auf Navigationsmustern wahrscheinlich als Nächstes braucht
- Überwachen Sie echte Nutzermetriken, um zu verifizieren, dass Splitting die tatsächlichen Ladezeiten verbessert

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verringert die initiale Seitenladezeit, indem nur der für die aktuelle Ansicht benötigte Code geladen wird
- Verbessert die Cache-Effizienz, weil unveränderte Chunks bei Updates nicht erneut heruntergeladen werden
- Ermöglicht inkrementelles Laden, das die Anwendung für Nutzer schneller erscheinen lässt
- Verringert den Speicherverbrauch auf ressourcenbeschränkten Geräten

**Kosten und Risiken:**
- Fügt der Build-Konfiguration und Modulstruktur Komplexität hinzu
- Könnte Ladeverzögerungen einführen, während zu neuen Abschnitten navigiert wird, die zusätzliche Chunks abrufen müssen
- Über-Splitting erzeugt zu viele kleine Netzwerkanfragen, was die Performance verschlechtern kann
- Legacy-Bundling-Konfigurationen könnten erhebliche Überarbeitung erfordern, um Code Splitting zu unterstützen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Single-Page-Anwendung für ein Versicherungsportal lud bei jedem Seitenaufruf ein 4,5-MB-JavaScript-Bundle, einschließlich Code für Agenten-Dashboards, Schadensmeldungsformulare und Berichtsdiagramme, auf die die meisten Nutzer nie zugriffen. Das Team führte routenbasiertes Code Splitting ein, verringerte das initiale Bundle auf 800 KB und lud zusätzliche Module bei Bedarf. Sie teilten außerdem die Diagrammbibliothek in einen lazy-geladenen Chunk auf, da nur der Berichtsabschnitt sie nutzte. Die durchschnittliche Seitenladezeit sank von 6 Sekunden auf 1,8 Sekunden bei typischen Verbindungen, und mobile Nutzer berichteten von einer dramatisch verbesserten Erfahrung.
