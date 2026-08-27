---
title: Virtualisierte Listen
description: Effiziente Darstellung großer Datenlisten durch virtuelle
  Scroll-Bereiche.
category:
- Performance
- Code
problems:
- slow-response-times-for-lists
- high-client-side-resource-consumption
- slow-application-performance
- memory-leaks
- high-resource-utilization-on-client
- inefficient-frontend-code
layout: solution
lang: de
en_slug: virtualized-lists
related_solutions:
- slug: pagination
  similarity: 0.7
- slug: performance-optimization
  similarity: 0.65
- slug: lazy-evaluation
  similarity: 0.65
- slug: lazy-loading
  similarity: 0.65
- slug: image-and-asset-optimization
  similarity: 0.6
- slug: predictive-loading
  similarity: 0.6
---

## Description

Virtualisierte Listen rendern nur die Zeilen, die derzeit innerhalb eines Scroll-Viewports sichtbar sind, plus einen kleinen Puffer, und recyceln denselben begrenzten Satz von DOM-Elementen, während der Nutzer scrollt, statt für jedes Element eines Datensatzes, der Zehntausende von Zeilen enthalten könnte, ein DOM-Element zu erstellen. Viele Legacy-Frontend-Komponenten datieren vor dieser Technik und rendern einfach jede Zeile einer Tabelle oder Liste bedingungslos, ein Ansatz, der für kleine Datensätze akzeptabel skaliert, aber katastrophal degradiert, während die zugrunde liegenden Daten wachsen — ein Muster, das in Legacy-Systemen häufig ist, die gebaut wurden, als Datenvolumina weit kleiner waren und niemand erwartete, dass der Datensatz schließlich die Größe erreichen würde, die er heute hat. Die Performance-Kosten sind an dem Punkt kein seltener Grenzfall, sondern ein routinemäßiges, reproduzierbares Einfrieren bei jedem Seitenaufruf, da der Browser eine enorme Anzahl von DOM-Knoten für eine Ansicht konstruieren, layouten und schließlich garbage-collecten muss, in der der Nutzer immer nur eine Handvoll davon gleichzeitig betrachten kann. Das Ersetzen des naiven Renderings durch eine Virtualisierungsbibliothek stellt Responsivität wieder her, indem die Anzahl der DOM-Elemente begrenzt und ungefähr konstant gehalten wird, unabhängig von der Datensatzgröße, auf Kosten zusätzlicher Rendering-Komplexität — besonders für Zeilen variabler Höhe — und dem Verlust bestimmter browsernativer Verhaltensweisen wie In-Page-Textsuche, die typischerweise durch ein explizites serverseitiges Suchfeature ersetzt werden muss, um dies auszugleichen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Listen- oder Tabellenkomponenten, die Hunderte oder Tausende von DOM-Elementen gleichzeitig rendern
- Ersetzen Sie traditionelles Listen-Rendering durch eine Virtualisierungsbibliothek (react-window, react-virtualized, Angular CDK Virtual Scroll oder ähnlich)
- Rendern Sie nur die sichtbaren Zeilen plus einen kleinen Puffer, und recyceln Sie DOM-Elemente, während der Nutzer scrollt
- Berechnen Sie Zeilenhöhen akkurat (fest oder variabel), um korrekte Scroll-Position und Scrollbar-Verhalten aufrechtzuerhalten
- Kombinieren Sie Virtualisierung mit serverseitiger Paginierung, sodass der Client nie den vollständigen Datensatz im Speicher halten muss
- Handhaben Sie Grenzfälle: Tastaturnavigation, Screenreader und Suche-innerhalb-der-Liste-Funktionalität

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Erhält flüssige Scroll-Performance unabhängig von der Listengröße
- Reduziert die DOM-Elementanzahl dramatisch, senkt Speicherverbrauch und verbessert die Rendering-Geschwindigkeit
- Ermöglicht die Anzeige von Datensätzen mit Zehntausenden von Elementen, die sonst unmöglich zu rendern wären
- Reduziert Garbage-Collection-Druck durch das Erstellen und Zerstören von DOM-Knoten

**Kosten und Risiken:**
- Fügt der Rendering-Logik Komplexität hinzu, besonders für Zeilen variabler Höhe
- Barrierefreiheit kann leiden, wenn Screenreader nicht auf Off-Screen-Elemente zugreifen können
- Die Browser-Suche (Strg+F) funktioniert nicht für Elemente, die derzeit nicht gerendert werden
- Die Scroll-Positions-Verwaltung wird komplex, wenn Listenelemente dynamisch eingefügt, entfernt oder in der Größe geändert werden
- Die Integration mit Legacy-DOM-manipulierendem Code könnte mit den Annahmen der Virtualisierungsbibliothek kollidieren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Bestandsverwaltungsanwendung renderte alle 50.000 Produkt-SKUs in einer einzigen HTML-Tabelle, was den Browser während des anfänglichen Renderns für mehrere Sekunden einfrieren ließ und über 1 GB Speicher verbrauchte. Das Team ersetzte die Tabelle durch react-window und renderte nur die 30 sichtbaren Zeilen plus einen 10-Zeilen-Puffer in jeder Richtung. Die anfängliche Renderzeit sank von 8 Sekunden auf 50 Millisekunden, und der Speicherverbrauch für die Liste sank auf unter 10 MB. Das Team fügte auch serverseitige Suche und Filterung hinzu, sodass Nutzer spezifische SKUs finden konnten, ohne durch die gesamte Liste zu scrollen, was den Verlust der browsernativen Textsuche ausglich.
