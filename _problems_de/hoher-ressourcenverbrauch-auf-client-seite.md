---
title: Hoher Ressourcenverbrauch auf Client-Seite
description: Client-Anwendungen verbrauchen übermäßig viel CPU oder Speicher, was
  zu träger Performance und schlechtem Nutzererlebnis führt.
category:
- Performance
related_problems:
- slug: high-resource-utilization-on-client
  similarity: 0.95
- slug: inefficient-frontend-code
  similarity: 0.75
- slug: slow-application-performance
  similarity: 0.65
- slug: algorithmic-complexity-problems
  similarity: 0.65
- slug: high-number-of-database-queries
  similarity: 0.6
- slug: resource-contention
  similarity: 0.6
solutions:
- user-centered-design
- api-calls-optimization
- asynchronous-operations
- browser-compatibility
- code-splitting
- compression
- image-and-asset-optimization
- lazy-evaluation
- lazy-loading
- pagination
- performance-budgets
- predictive-prefetching
- progressive-loading
- tree-shaking
- virtualized-lists
- mobile-first-design
- performance-optimization
- responsive-design
layout: problem
lang: de
en_slug: high-client-side-resource-consumption
---

## Description
Hoher Ressourcenverbrauch auf Client-Seite kann zu einem schlechten Nutzererlebnis führen. Dies kann sich als träge Benutzeroberfläche, hoher Akkuverbrauch auf mobilen Geräten oder ein allgemeines Gefühl von Unresponsivität äußern. Verbreitete Ursachen für hohen Ressourcenverbrauch sind ineffizientes JavaScript, große, nicht optimierte Assets und übermäßige DOM-Manipulation. Ein Fokus auf Client-seitige Performance ist essenziell, um ein schnelles und reaktionsfreudiges Nutzererlebnis zu schaffen.

## Indicators ⟡
- Die Anwendung ist selbst auf einem leistungsstarken Gerät langsam.
- Die Anwendung entlädt den Akku des mobilen Geräts.
- Der Lüfter des Computers läuft auf Hochtouren bei der Nutzung der Anwendung.
- Es kommen Beschwerden von Nutzern über langsame Performance.

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Übermäßige CPU- und Speichernutzung auf dem Client lässt die Anwendung träge und unresponsiv erscheinen.
- [Negatives Nutzerfeedback](negatives-nutzerfeedback.md)
<br/>  Nutzer beklagen sich über langsame Performance, heiße Geräte und Akkuverbrauch, verursacht durch ressourcenhungrige Client-Anwendungen.
- [Nutzerfrustration](nutzerfrustration.md)
<br/>  Nutzer werden unzufrieden, wenn die Anwendung ihr Gerät langsam oder heiß macht oder schnell den Akku entlädt.

## Causes ▼

- [Ineffizienter Frontend-Code](ineffizienter-frontend-code.md)
<br/>  Nicht optimiertes JavaScript, übermäßige DOM-Manipulation und komplexe CSS-Animationen verbrauchen übermäßig viel Client-CPU und -Speicher.
- [Speicherlecks](speicherlecks.md)
<br/>  Client-seitige Speicherlecks durch nicht freigegebene DOM-Elemente oder Event-Listener verursachen kontinuierlich wachsenden Speicherverbrauch.
- [Unsachgemäße Verwaltung von Event-Listenern](unsachgemaesse-verwaltung-von-event-listenern.md)
<br/>  Event-Listener, die nie entfernt werden, häufen sich über die Zeit an und verbrauchen Speicher und CPU-Ressourcen auf dem Client.
- [Ineffizienter Code](ineffizienter-code.md)
<br/>  Rechenintensiver Client-seitiger Code, wie große Schleifen oder komplexe Rendering-Logik, verbraucht übermäßig CPU-Ressourcen.

## Detection Methods ○

- **Browser-Entwicklerwerkzeuge:** Nutzung der Performance-, Speicher- und Netzwerk-Tabs in Browser-Entwicklerwerkzeugen zum Profiling von CPU-Nutzung, Speicherverbrauch und Netzwerkaktivität.
- **Real User Monitoring (RUM):** RUM-Werkzeuge können Performance-Metriken aus tatsächlichen Nutzersitzungen sammeln, einschließlich CPU- und Speichernutzung.
- **Gerätespezifisches Monitoring:** Nutzung von Betriebssystem-Werkzeugen (z. B. Activity Monitor auf macOS, Task-Manager auf Windows, Android Studio Profiler, Xcode Instruments) zur Überwachung der Ressourcennutzung.
- **Code-Review:** Achten auf verbreitete Antipatterns wie große Schleifen, übermäßige Event-Listener oder nicht optimierte Rendering-Logik.

## Examples
Eine Single-Page-Anwendung (SPA) wird sehr langsam, nachdem ein Nutzer lange mit ihr interagiert hat. Profiling zeigt ein Speicherleck, bei dem alte DOM-Elemente nicht der Garbage Collection unterzogen werden, was zu kontinuierlichem Speicherwachstum führt. In einem anderen Fall nutzt eine Website ein großes, nicht optimiertes Hintergrundvideo auf ihrer Startseite. Auf mobilen Geräten führt dies dazu, dass der Browser erheblich CPU und Akku verbraucht, was das Telefon heiß macht und den Akku schnell entlädt. Dieses Problem wird mit dem Aufstieg komplexer Webanwendungen und mobiler Apps, die direkt auf Nutzergeräten laufen, zunehmend verbreitet. Die Optimierung der Client-seitigen Performance ist entscheidend für die Bereitstellung eines reibungslosen und angenehmen Nutzererlebnisses.
