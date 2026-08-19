---
title: Paginierung
description: Laden großer Datenausgaben in kleinere, handhabbare Abschnitte.
category:
- Performance
- Code
problems:
- slow-response-times-for-lists
- high-client-side-resource-consumption
- slow-database-queries
- memory-leaks
- unbounded-data-growth
- high-number-of-database-queries
- slow-application-performance
- graphql-complexity-issues
- lazy-loading
- unbounded-data-structures
layout: solution
lang: de
en_slug: pagination
related_solutions:
- slug: lazy-loading
  similarity: 0.75
- slug: lazy-evaluation
  similarity: 0.75
- slug: performance-optimization
  similarity: 0.7
- slug: api-calls-optimization
  similarity: 0.7
- slug: data-partitioning
  similarity: 0.7
- slug: predictive-loading
  similarity: 0.7
---

## Description

Paginierung teilt eine große Ergebnismenge in kleinere, begrenzte Abschnitte auf, die inkrementell angefordert und gerendert werden, statt einen gesamten Datensatz in einer einzigen Antwort zurückzugeben. Legacy-Systeme sind für diesen nachträglichen Einbau besonders anfällig: Endpunkte und Bildschirme, die ursprünglich gebaut wurden, als eine Tabelle ein paar hundert Zeilen hatte, werden still zu Belastungen, sobald diese Tabelle auf Millionen wächst, was den Speicher sowohl auf Server als auch Client erschöpft und zuvor sofortige Abfragen zu mehrsekündigen macht. Die Einführung von Paginierung — sei es einfache Offset-basierte Seitenaufteilung oder der besser skalierbare Cursor-basierte Ansatz — begrenzt sowohl die Abfragekosten als auch die Antwortgröße, muss aber sorgfältig zu bestehenden APIs hinzugefügt werden, da Konsumenten, die derzeit eine vollständige Ergebnismenge erwarten, brechen können, sobald Paginierung verpflichtend statt optional wird.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Endpunkte und Bildschirme, die unbegrenzte Ergebnismengen zurückgeben, insbesondere solche, die über ihre ursprünglich erwartete Größe hinausgewachsen sind
- Wählen Sie eine Paginierungsstrategie: Offset-basiert für Einfachheit, Cursor-basiert für Konsistenz bei großen oder sich häufig ändernden Datensätzen
- Fügen Sie LIMIT- und OFFSET-Klauseln (oder Keyset-basierte) zu den zugrunde liegenden Datenbankabfragen hinzu
- Implementieren Sie Paginierungsparameter in der API-Schicht mit sinnvollen Standardwerten und maximalen Seitengrößen
- Aktualisieren Sie das Frontend, um Seitensteuerungen oder unendliches Scrollen mit progressivem Laden anzuzeigen
- Stellen Sie sicher, dass die Sortierreihenfolge deterministisch ist, um zu verhindern, dass Elemente auf mehreren Seiten erscheinen oder übersprungen werden
- Rüsten Sie bestehende API-Konsumenten schrittweise nach, indem Sie Paginierung mit einem Standardlimit optional machen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verhindert Speichererschöpfung auf Server und Client, wenn Datensätze über die ursprünglichen Erwartungen hinauswachsen
- Reduziert Datenbankabfragezeit und Netzwerkübertragungsgröße
- Verbessert die wahrgenommene Performance durch schnelle Anzeige erster Ergebnisse
- Begrenzt den Explosionsradius langsamer Abfragen auf kleinere Ergebnismengen

**Kosten und Risiken:**
- Offset-basierte Paginierung wird bei hohen Offsets auf großen Tabellen langsam
- Das Hinzufügen von Paginierung zu bestehenden APIs kann Konsumenten brechen, die vollständige Ergebnismengen erwarten
- Cursor-basierte Paginierung ist komplexer zu implementieren und erfordert stabile Sortierschlüssel
- Nutzer finden spezifische Elemente möglicherweise nicht leicht, wenn Suche oder Filterung nicht ebenfalls verbessert werden
- Legacy-Berichte, die auf der gleichzeitigen Verarbeitung aller Datensätze beruhen, erfordern Überarbeitung

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine ursprünglich für ein paar hundert Tickets pro Mitarbeiter konzipierte Kundensupport-Anwendung bediente nun Teams, die Zehntausende offener Fälle verwalteten. Der Ticketlisten-Endpunkt gab alle Tickets in einer einzigen Antwort zurück, was zu Browser-Tab-Abstürzen und API-Timeouts während Spitzenzeiten führte. Das Team fügte Cursor-basierte Paginierung hinzu, wobei der Ticket-Erstellungszeitstempel als Cursor diente, mit einer Standardseitengröße von 50. Kombiniert mit serverseitiger Filterung reduzierte dies die durchschnittliche API-Antwortzeit von 12 Sekunden auf 200 Millisekunden und beseitigte die Browser-Speicherprobleme vollständig.
