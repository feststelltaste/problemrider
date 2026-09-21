---
title: Tree Shaking
description: Entfernung ungenutzten Codes während des Builds.
category:
- Code
- Performance
problems:
- high-client-side-resource-consumption
- slow-application-performance
- uncontrolled-codebase-growth
- feature-bloat
- inefficient-frontend-code
- gradual-performance-degradation
layout: solution
lang: de
en_slug: tree-shaking
related_solutions:
- slug: code-splitting
  similarity: 0.8
- slug: image-and-asset-optimization
  similarity: 0.8
- slug: strategic-code-deletion
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.75
- slug: lazy-evaluation
  similarity: 0.75
- slug: compression
  similarity: 0.7
---

## Description

Tree Shaking ist eine Build-Zeit-Optimierung, die von Modul-Bundlern durchgeführt wird und die den Import- und Export-Graphen einer Codebasis statisch analysiert und jeglichen Code entfernt, auf den nie tatsächlich verwiesen wird, sodass das ausgelieferte Bundle nur das enthält, was die Anwendung nutzt, statt alles, was eine Abhängigkeit zufällig bereitstellt. Es verlässt sich auf die statische, analysierbare Struktur von ES-Modulen, um Erreichbarkeit zur Build-Zeit zu bestimmen, weshalb Legacy-CommonJS-Code — mit seinen dynamischen `require()`-Aufrufen, die nicht immer ohne Programmausführung aufgelöst werden können — es häufig unterläuft und vor der Optimierung konvertiert werden muss. In Legacy-Frontend-Codebasen zählt dies, weil die Bundle-Größe über Jahre monoton wächst: ganze Utility-Bibliotheken werden für eine Handvoll Funktionen importiert, deaktivierte Features bleiben gebündelt, weil niemand ihre Imports entfernte, und Barrel-Dateien re-exportieren alles wahllos, nichts davon zeigt sich als funktionaler Fehler, aber alles davon besteuert still jeden Seitenaufruf. Tree Shaking adressiert dies, ohne dass jemand toten Code manuell Pfad für Pfad aufspüren und löschen muss; stattdessen entfernt der Build-Prozess selbst, was statische Analyse als unerreichbar beweist, gegeben genug strukturelle Bereinigung (ES-Module, nebenwirkungsfreie Paket-Markierungen, Vermeidung übermäßig breiter Barrel-Exports), um die Analyse effektiv zu machen. Da es vollständig innerhalb der Build-Pipeline operiert, kann es inkrementell zusammen mit anderer Modernisierungsarbeit übernommen werden und liefert messbare Seitenladeverbesserungen, ohne eine Neuschreibung des Laufzeitverhaltens der Anwendung zu erfordern.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Konfigurieren Sie Modul-Bundler (Webpack, Rollup, esbuild), um Dead-Code-Eliminierung während des Build-Prozesses durchzuführen
- Konvertieren Sie Legacy-CommonJS-Module zu ES-Modulen, um statische Analyse von Import-/Export-Abhängigkeiten zu ermöglichen
- Markieren Sie Pakete und Module als nebenwirkungsfrei in package.json, um aggressiveres Tree Shaking zu erlauben
- Prüfen Sie Bundle-Inhalte mit Visualisierungswerkzeugen (webpack-bundle-analyzer), um große ungenutzte Abhängigkeiten zu identifizieren
- Ersetzen Sie monolithische Utility-Bibliotheken durch modulare Alternativen, die Pro-Funktion-Imports unterstützen
- Refaktorieren Sie Barrel-Dateien (index.js-Re-Exports), die Tree Shaking daran hindern, ungenutzte Exports zu identifizieren
- Fügen Sie Bundle-Größen-Prüfungen zur CI-Pipeline hinzu, um Regression zu verhindern

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert die JavaScript-Bundle-Größe und verbessert direkt die Seitenladezeiten
- Entfernt toten Code, der die tatsächlich genutzte Codebasis verschleiert
- Verringert clientseitigen Speicherverbrauch und Parsing-Zeit
- Kann inkrementell zusammen mit anderen Modernisierungsanstrengungen implementiert werden

**Kosten und Risiken:**
- Legacy-Code mit Nebenwirkungen in der Modulinitialisierung kann brechen, wenn er tree-geshaked wird
- Dynamische Imports und require()-Aufrufe können nicht statisch analysiert werden und könnten fälschlicherweise entfernt werden
- Erfordert Migration von CommonJS zu ES-Modulen, was in großen Codebasen störend sein kann
- Die Build-Konfigurationskomplexität steigt mit Tree-Shaking-Regeln und Ausnahmen
- Manche Bibliotheken sind nicht tree-shakeable, was Ersatz oder manuellen Ausschluss erfordert

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine mit Angular.js gebaute Legacy-Single-Page-Anwendung hatte über fünf Jahre ein 4,2-MB-JavaScript-Bundle angesammelt. Die Bundle-Analyse offenbarte, dass ein vollständiger Lodash-Import 600 KB beitrug, obwohl nur 12 Funktionen genutzt wurden, und mehrere Feature-Module, die in der Konfiguration deaktiviert worden waren, immer noch eingeschlossen waren. Das Team wechselte zu lodash-es mit Pro-Funktion-Imports, konvertierte Schlüsselmodule zu ES-Modul-Syntax und aktivierte Webpacks Tree Shaking. Das Produktions-Bundle sank auf 1,8 MB, was die anfängliche Seitenladezeit von 6 Sekunden auf 2,5 Sekunden bei typischen Verbindungen reduzierte.
