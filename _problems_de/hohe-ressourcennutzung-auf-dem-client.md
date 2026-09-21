---
title: Hohe Ressourcennutzung auf dem Client
description: Client-Anwendungen könnten übermäßig viel CPU oder Speicher verbrauchen,
  was zu einem schlechten Nutzererlebnis führt, besonders auf weniger leistungsstarken
  Geräten.
category:
- Performance
- Requirements
related_problems:
- slug: high-client-side-resource-consumption
  similarity: 0.95
- slug: inefficient-frontend-code
  similarity: 0.75
- slug: slow-application-performance
  similarity: 0.65
- slug: resource-contention
  similarity: 0.65
- slug: high-number-of-database-queries
  similarity: 0.6
- slug: high-database-resource-utilization
  similarity: 0.6
solutions:
- user-centered-design
- image-and-asset-optimization
- lazy-loading
- virtualized-lists
- performance-budgets
- profiling
- code-splitting
- progressive-loading
- performance-measurements
layout: problem
lang: de
en_slug: high-resource-utilization-on-client
---

## Description
Hohe Ressourcennutzung auf der Client-Seite kann zu einem schlechten Nutzererlebnis führen. Dies kann sich als träge Benutzeroberfläche, hoher Akkuverbrauch auf mobilen Geräten oder ein allgemeines Gefühl von Unresponsivität äußern. Verbreitete Ursachen für hohe Ressourcennutzung sind ineffizientes JavaScript, große, nicht optimierte Assets und übermäßige DOM-Manipulation. Ein Fokus auf Client-seitige Performance ist essenziell, um ein schnelles und reaktionsfreudiges Nutzererlebnis zu schaffen.

## Indicators ⟡
- Die Anwendung ist selbst auf einem leistungsstarken Gerät langsam.
- Die Anwendung entlädt den Akku des mobilen Geräts.
- Der Lüfter des Computers läuft auf Hochtouren bei der Nutzung der Anwendung.
- Es kommen Beschwerden von Nutzern über langsame Performance.

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Übermäßiger Client-seitiger Ressourcenverbrauch lässt die Anwendungs-UI für Nutzer träge und unresponsiv werden.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Nutzer erleben schlechte Performance, Akkuverbrauch und Geräteüberhitzung, was zu Frustration und Abwanderung führt.

## Causes ▼

- [Ineffizienter Frontend-Code](ineffizienter-frontend-code.md)
<br/>  Schlecht optimiertes JavaScript, übermäßige DOM-Manipulation und unnötige Re-Renders verbrauchen übermäßig Client-Ressourcen.
- [Speicherlecks](speicherlecks.md)
<br/>  Nicht freigegebener Speicher durch unsachgemäß verwaltete Objekte und Event-Listener verbraucht schrittweise verfügbare Client-Ressourcen.
- [Unsachgemäße Verwaltung von Event-Listenern](unsachgemaesse-verwaltung-von-event-listenern.md)
<br/>  Angehäufte, nicht entfernte Event-Listener verbrauchen Speicher und führen unnötigen Code aus, was CPU- und Speichernutzung erhöht.

## Detection Methods ○

- **Browser-Entwicklerwerkzeuge:** Nutzung der Performance-, Speicher- und Netzwerk-Tabs in Browser-Entwicklerwerkzeugen zum Profiling der Client-seitigen Aktivität.
- **Real User Monitoring (RUM):** RUM-Werkzeuge können Client-seitige Performance-Metriken von tatsächlichen Nutzern sammeln.
- **Geräte-Monitoring-Werkzeuge:** Nutzung von Betriebssystem-Werkzeugen (z. B. Activity Monitor auf macOS, Task-Manager auf Windows, Android Studio Profiler) zur Überwachung von CPU- und Speichernutzung der Client-Anwendung.
- **Nutzerfeedback:** Beachtung von Nutzerbeschwerden über Performance, Akkulaufzeit oder Geräteüberhitzung.

## Examples
Eine komplexe Webanwendung mit vielen interaktiven Elementen wird sehr langsam und lässt den Lüfter des Laptops des Nutzers hochdrehen. Profiling mit Browser-Entwicklerwerkzeugen zeigt, dass eine JavaScript-Funktion ständig einen großen Teil des DOM in einer ineffizienten Schleife neu rendert. In einem anderen Fall hat ein mobiles Spiel nicht optimierte Texturen und Modelle. Wenn es auf einem älteren Telefon gespielt wird, ruckelt das Spiel häufig und lässt das Gerät sehr heiß werden, was den Akku schnell entlädt. Dieses Problem wird zunehmend verbreitet, während Anwendungen funktionsreicher werden und auf einer breiteren Vielfalt von Geräten laufen. Die Optimierung der Client-seitigen Performance ist entscheidend für ein gutes Nutzererlebnis, besonders auf mobilen und weniger leistungsstarken Geräten.
