---
title: Prädiktives Prefetching
description: Laden wahrscheinlich benötigter Inhalte, abgeleitet aus
  aktueller Nutzung.
category:
- Performance
problems:
- slow-application-performance
- high-api-latency
- network-latency
- poor-user-experience-ux-design
- user-frustration
- high-client-side-resource-consumption
layout: solution
lang: de
en_slug: predictive-prefetching
related_solutions:
- slug: predictive-loading
  similarity: 0.85
- slug: progressive-loading
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.7
- slug: lazy-evaluation
  similarity: 0.7
- slug: code-splitting
  similarity: 0.7
- slug: image-and-asset-optimization
  similarity: 0.65
---

## Description

Prädiktives Prefetching erweitert dieselbe Idee wie prädiktives Laden hinunter auf die Ebene einzelner Interaktionssignale — Mausbewegung, Scroll-Position, Hover-Status, Navigationshistorie — und nutzt diese niedrigschwelligen Hinweise, um das Laden von Route-Bundles, API-Antworten oder statischen Assets für den spezifischen nächsten Bildschirm auszulösen, auf den ein Nutzer offenbar zusteuert, oft durch Service Worker, die während der Browser-Leerlaufzeit arbeiten. Da es auf dieser feineren Granularität operiert, kann es auf ein bestehendes Legacy-Frontend aufgeschichtet werden, ohne tiefgreifende architektonische Änderungen, was einem langsamen Legacy-Editor oder einer Inhaltsseite ein nahezu sofortiges Gefühl gibt, indem seine Ressourcen bereits im Browser zwischengespeichert sind, wenn der Nutzer tatsächlich klickt. Dies macht es attraktiv für Legacy-Modernisierungsanstrengungen unter Zeit- oder Budgetdruck, wo eine vollständige Neuschreibung einer langsamen Seite nicht machbar ist, aber eine begrenzte Frontend-Erweiterung schon. Die Technik hängt von hochsicheren Heuristiken und einem gedeckelten Prefetch-Budget ab, um den Fehlerfall zu vermeiden, bei dem spekulative Anfragen für unwahrscheinliche nächste Aktionen Bandbreite verschwenden und unnötige Last zu einem bereits belasteten Legacy-Backend hinzufügen. Es interagiert auch schlecht mit Systemen, die strenges Rate-Limiting oder kurzlebige Authentifizierungstoken haben, da spekulative Anfragen dasselbe Kontingent und Token-Budget verbrauchen wie Anfragen, die der Nutzer tatsächlich beabsichtigt, sodass die Prefetching-Logik sich dieser Beschränkungen bewusst sein muss, statt unabhängig von ihnen zu operieren.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Verfolgen Sie Nutzerinteraktionsmuster (Mausbewegungen, Scroll-Position, Navigationshistorie), um kommende Ressourcenbedürfnisse vorherzusagen
- Implementieren Sie Prefetching für Routenebenen-Code-Bundles, wenn der Nutzer über ein Navigationselement hovert oder sich ihm nähert
- Verwenden Sie Service Worker, um API-Antworten für wahrscheinliche nächste Aktionen während der Browser-Leerlaufzeit vorzuladen
- Wenden Sie heuristische Regeln basierend auf Domänenwissen an (z. B. nach Ansicht einer Produktliste die Top-Produktdetails vorladen)
- Beschränken Sie Prefetching auf hochsichere Vorhersagen und deckeln Sie das gesamte Prefetch-Budget, um Ressourcenverschwendung zu vermeiden
- Messen Sie Cache-Trefferquoten für vorgeladenen Inhalt, um die Vorhersagegenauigkeit über die Zeit zu validieren und zu justieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt Ladeverzögerungen für korrekt vorhergesagte Navigationen und schafft nahezu sofortige Übergänge
- Nutzt Leerlaufzeit und Bandbreite, die sonst ungenutzt blieben
- Kann auf bestehende Legacy-Frontends aufgeschichtet werden, ohne tiefgreifende Umgestaltung
- Verbessert Nutzerzufriedenheitskennzahlen durch Reduzierung der Zeit bis zum Inhalt

**Kosten und Risiken:**
- Verschwendete Bandbreite für fehlvorhergesagte Prefetches, besonders bei getakteten Verbindungen
- Erhöhte Serverlast durch spekulative Anfragen, die möglicherweise nie genutzt werden
- Komplexität der Pflege der Vorhersagelogik neben dem Anwendungscode
- Vorgeladene Daten können veralten, wenn sich zugrunde liegende Daten häufig ändern
- Kann mit Rate-Limiting oder Authentifizierungstoken-Verwaltung interferieren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein von Journalisten genutztes Legacy-Content-Management-System hatte einen langsamen Dokumenteneditor, der 4 Sekunden zum Laden benötigte, aufgrund des Abrufens von Vorlagen, Stylesheets und aktueller Dokumenthistorie. Das Team fügte eine Prefetching-Schicht hinzu, die begann, Editor-Ressourcen zu laden, sobald der Nutzer zur Dokumentenliste navigierte, da 90 Prozent der Dokumentenlisten-Ansichten vom Öffnen eines Dokuments zur Bearbeitung gefolgt wurden. Zu dem Zeitpunkt, an dem der Nutzer auf ein Dokument klickte, war die Editor-Hülle bereits im Browser zwischengespeichert, was die wahrgenommene Ladezeit auf unter 500 Millisekunden reduzierte.
