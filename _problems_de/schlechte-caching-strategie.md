---
title: Schlechte Caching-Strategie
description: Daten, die gecacht werden könnten, werden bei jeder Anfrage von der
  Quelle abgerufen, was unnötigen Overhead hinzufügt und die Latenz erhöht.
category:
- Performance
related_problems:
- slug: slow-database-queries
  similarity: 0.7
- slug: cache-invalidation-problems
  similarity: 0.7
- slug: high-api-latency
  similarity: 0.7
- slug: slow-application-performance
  similarity: 0.7
- slug: n-plus-one-query-problem
  similarity: 0.7
- slug: high-number-of-database-queries
  similarity: 0.7
solutions:
- caching-strategy
- distributed-caching
- performance-measurements
- profiling
- monitoring
- performance-modeling
- materialized-views
- load-testing
- continuous-performance-monitoring
- index-lifecycle-management
layout: problem
lang: de
en_slug: poor-caching-strategy
---

## Description
Eine schlechte Caching-Strategie kann so schlimm sein wie gar kein Caching. Dieses Problem umfasst eine Reihe von Themen, von zu viel oder zu wenig gecachten Daten über die Nutzung unangemessener Cache-Eviction-Richtlinien bis hin zum Fehlen einer klaren Strategie für Cache-Invalidierung. Eine ineffektive Caching-Strategie kann dazu führen, dass Nutzern veraltete Daten geliefert werden, oder zu einer niedrigen Cache-Trefferquote, die die Performance-Vorteile des Cachings zunichtemacht. Eine gut designte Caching-Strategie ist eine kritische Komponente jeder Hochleistungsanwendung.

## Indicators ⟡
- Die Anwendung ist langsam, obwohl die Datenbank nicht stark belastet ist.
- Die Anwendung macht viele unnötige Anfragen an die Datenbank oder andere Services.
- Die Cache-Trefferquote ist niedrig.
- Nutzer sehen veraltete Daten.

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Das wiederholte Abrufen von Daten, die gecacht werden könnten, fügt jeder Anfrage unnötige Latenz hinzu, was die Anwendung träge wirken lässt.
- [Hohe API-Latenz](hohe-api-latenz.md)
<br/>  API-Endpunkte, die Daten bei jeder Anfrage von der Quelle abrufen statt aus dem Cache zu bedienen, zeigen unnötig hohe Antwortzeiten.
- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Redundante Datenbankabfragen durch fehlendes oder ineffektives Caching erzeugen exzessive Last auf Datenbankservern.
- [Hohe Anzahl an Datenbankabfragen](hohe-anzahl-an-datenbankabfragen.md)
<br/>  Ohne Caching werden dieselben Daten wiederholt aus der Datenbank abgefragt, was die Gesamtzahl der Datenbankanfragen aufbläht.
- [Langsame Antwortzeiten für Listen](langsame-antwortzeiten-fuer-listen.md)
<br/>  Listenseiten, die Daten aus mehreren Quellen aggregieren, sind besonders von schlechtem Caching betroffen, da jedes Element separate ungecachte Abfragen auslösen kann.

## Causes ▼

- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Erfahrung in Performance-Optimierung erkennen möglicherweise keine Caching-Möglichkeiten oder wissen nicht, wie man effektive Strategien implementiert.
- [Rapid Prototyping wird zu Produktion](rapid-prototyping-wird-zu-produktion.md)
<br/>  Prototyp-Code, der Caching der Einfachheit halber übersprang, gelangt in Produktion, ohne dass die Caching-Schicht später hinzugefügt wird.
- [Implementierung beginnt ohne Design](implementierung-beginnt-ohne-design.md)
<br/>  Der Beginn der Entwicklung ohne vorheriges Design bedeutet, dass Caching-Strategien nicht als Teil der Architektur berücksichtigt werden.

## Detection Methods ○

- **Netzwerk-Monitoring:** Analyse des Netzwerkverkehrs, um zu sehen, ob dieselben Daten wiederholt abgerufen werden.
- **Backend-System-Metriken:** Überwachung der Last auf Datenbanken oder anderen Services zur Identifikation repetitiver Abfragen.
- **Cache-Hit-/Miss-Verhältnisse:** Wenn eine Caching-Lösung vorhanden ist, Überwachung ihres Hit-/Miss-Verhältnisses zur Bewertung ihrer Effektivität.
- **Anwendungs-Profiling:** Nutzung von Profiling-Werkzeugen zur Identifikation von Zeit, die für das Abrufen von Daten aus der Quelle aufgewendet wird, die aus einem Cache hätten bedient werden können.
- **HTTP-Header-Analyse:** Für Webanwendungen Untersuchung von HTTP-Antwort-Headern, um sicherzustellen, dass ordentliche Cache-Control-Direktiven gesendet werden.

## Examples
Eine E-Commerce-Website zeigt Produktkategorien an. Jedes Mal, wenn ein Nutzer zur Startseite navigiert, wird die Liste der Kategorien direkt aus der Datenbank abgerufen, obwohl sie sich selten ändert. Dies fügt der Datenbank unnötige Last hinzu und erhöht die Seitenladezeit. In einem anderen Fall ruft ein Microservice Konfigurationsdaten von einem zentralen Konfigurationsservice bei jedem API-Aufruf ab. Diese Daten ändern sich selten, aber es gibt keinen lokalen Cache, was zu ständigen Netzwerkaufrufen und erhöhter Latenz führt. Dieses Problem wird oft in den anfänglichen Entwicklungsphasen übersehen, wird aber kritisch, während eine Anwendung skaliert. Eine gut implementierte Caching-Strategie kann Latenz und Last auf Backend-Systemen erheblich reduzieren.
