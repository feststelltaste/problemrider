---
title: Ineffizienter Frontend-Code
description: Nicht optimiertes JavaScript, übermäßige DOM-Manipulation oder komplexe
  CSS-Animationen, die rechnerisch teuer sind.
category:
- Code
- Performance
related_problems:
- slug: high-resource-utilization-on-client
  similarity: 0.75
- slug: high-client-side-resource-consumption
  similarity: 0.75
- slug: inefficient-code
  similarity: 0.75
- slug: slow-application-performance
  similarity: 0.7
- slug: slow-database-queries
  similarity: 0.65
- slug: high-api-latency
  similarity: 0.6
solutions:
- user-centered-design
- browser-compatibility
- code-splitting
- image-and-asset-optimization
- tree-shaking
- virtualized-lists
- performance-budgets
- profiling
- performance-measurements
- lazy-loading
- continuous-performance-monitoring
layout: problem
lang: de
en_slug: inefficient-frontend-code
---

## Description
Ineffizienter Frontend-Code kann eine erhebliche Auswirkung auf das Nutzererlebnis haben. Dies kann sich als langsam ladende Seite, träge Benutzeroberfläche oder hoher Ressourcenverbrauch auf der Maschine des Kunden äußern. Verbreitete Ursachen ineffizienten Frontend-Codes sind große, nicht optimierte Assets, übermäßige DOM-Manipulation und fehlendes ordentliches Caching. Ein Fokus auf Frontend-Performance ist essenziell, um ein schnelles und reaktionsfreudiges Nutzererlebnis zu schaffen.

## Indicators ⟡
- Die Anwendung ist selbst auf einem leistungsstarken Gerät langsam.
- Die Anwendung entlädt den Akku des mobilen Geräts.
- Der Lüfter des Computers läuft auf Hochtouren bei der Nutzung der Anwendung.
- Es kommen Beschwerden von Nutzern über langsame Performance.

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Nicht optimierter Frontend-Code verursacht träge Seitenladezeiten und UI-Interaktionen, die für Endnutzer sichtbar sind.
- [Hoher Ressourcenverbrauch auf Client-Seite](hoher-ressourcenverbrauch-auf-client-seite.md)
<br/>  Ineffiziente DOM-Manipulation und nicht optimiertes JavaScript verbrauchen übermäßig CPU und Speicher auf Nutzergeräten.
- [Hohe Ressourcennutzung auf dem Client](hohe-ressourcennutzung-auf-dem-client.md)
<br/>  Rechnerisch teure Frontend-Operationen verursachen hohe CPU-Nutzung, Akkuverbrauch und Geräteerwärmung auf Client-Maschinen.

## Causes ▼

- [Ineffizienter Code](ineffizienter-code.md)
<br/>  Allgemeine Praktiken ineffizienten Codes übertragen sich auf die Frontend-Entwicklung und produzieren nicht optimierte Rendering-Logik und verschwenderische Berechnungen.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit Browser-Rendering-Pipelines und Performance-Optimierung nicht vertraut sind, schaffen ineffizienten Frontend-Code.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Ohne auf Performance fokussierte Reviews bleiben Frontend-Antipatterns wie übermäßige DOM-Manipulation unentdeckt.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Allgemeine Codequalitätsprobleme äußern sich als nicht optimierte Rendering-Logik, redundante Berechnungen und verschwenderische Ressourcennutzung im Frontend.

## Detection Methods ○

- **Browser-Entwicklerwerkzeuge:** Nutzung der Performance-, Speicher- und Netzwerk-Tabs in Browser-Entwicklerwerkzeugen zum Profiling von CPU-Nutzung, Speicherverbrauch und Rendering-Performance.
- **Lighthouse/PageSpeed Insights:** Nutzung dieser Werkzeuge für automatisierte Audits und Vorschläge zur Frontend-Performance.
- **Real User Monitoring (RUM):** RUM-Werkzeuge können Client-seitige Performance-Metriken aus tatsächlichen Nutzersitzungen sammeln.
- **Code-Review:** Achten auf verbreitete Antipatterns wie direkte DOM-Manipulation in Schleifen, komplexe CSS-Selektoren oder große JavaScript-Dateien.
- **Web Vitals:** Überwachung der Core Web Vitals (LCP, FID, CLS), um das Nutzererlebnis zu verstehen.

## Examples
Eine Webanwendung zeigt eine große Tabelle mit Tausenden von Zeilen an. Jedes Mal, wenn ein Nutzer die Tabelle sortiert oder filtert, wird das gesamte DOM neu gerendert, was ein merkliches Einfrieren der UI verursacht. Die Implementierung einer virtualisierten Listen- oder Tabellenkomponente würde die Performance erheblich verbessern. In einem anderen Fall ist eine JavaScript-Funktion dafür zuständig, einen Zähler auf dem Bildschirm jede Sekunde zu aktualisieren. Statt den Textinhalt eines einzelnen DOM-Elements direkt zu aktualisieren, erzeugt sie das gesamte Element neu und hängt es an das DOM an, was zu ständigen Reflows und hoher CPU-Nutzung führt. Mit der zunehmenden Komplexität von Webanwendungen und der Nachfrage nach reichhaltigen Nutzererlebnissen ist effizienter Frontend-Code von größter Bedeutung. Dieses Problem ist verbreitet in Anwendungen, die organisch ohne starken Fokus auf Performance-Optimierung gewachsen sind, oder wo Entwicklern tiefes Wissen über Browser-Rendering-Pipelines fehlt.
