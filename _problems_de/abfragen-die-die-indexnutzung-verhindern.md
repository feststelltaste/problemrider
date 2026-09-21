---
title: Abfragen, die die Indexnutzung verhindern
description: Die Art und Weise, wie eine Abfrage geschrieben ist, kann verhindern,
  dass die Datenbank einen verfügbaren Index nutzt, was langsamere vollständige
  Tabellenscans oder weniger effiziente Indexscans erzwingt.
category:
- Performance
related_problems:
- slug: inefficient-database-indexing
  similarity: 0.75
- slug: unused-indexes
  similarity: 0.7
- slug: incorrect-index-type
  similarity: 0.7
- slug: database-query-performance-issues
  similarity: 0.65
- slug: index-fragmentation
  similarity: 0.6
- slug: slow-database-queries
  similarity: 0.55
solutions:
- query-optimization-process
- static-code-analysis
- performance-measurements
- code-reviews
- profiling
- continuous-performance-monitoring
- data-modeling
- index-lifecycle-management
layout: problem
lang: de
en_slug: queries-that-prevent-index-usage
---

## Description
Selbst wenn angemessene Indizes existieren, können bestimmte Abfragemuster verhindern, dass die Datenbank sie effektiv nutzt, was zu langsamer Performance führt. Dies kann geschehen, wenn Funktionen auf indizierte Spalten angewendet werden, wenn Datentypen nicht übereinstimmen oder wenn der Abfrageoptimierer anderweitig nicht erkennen kann, dass ein Index die Abfrage erfüllen könnte. Das Schreiben „indexfreundlicher" Abfragen ist eine entscheidende Fähigkeit für Entwickler, die mit Datenbanken arbeiten, da es einen dramatischen Einfluss auf die Anwendungsperformance haben kann.

## Indicators ⟡
- Abfragen sind langsam, obwohl sie einen Index nutzen.
- Die Datenbank nutzt keinen Index, den Sie erwarten würden.
- Die Datenbank nutzt einen vollständigen Tabellenscan, obwohl ein Index verfügbar ist.
- Die Datenbank nutzt einen weniger effizienten Index als erwartet.

## Symptoms ▲

- [Langsame Datenbankabfragen](langsame-datenbankabfragen.md)
<br/>  Abfragen, die Indizes umgehen, erzwingen vollständige Tabellenscans, was direkt langsame Abfrageausführungszeiten verursacht.
- [Performance-Probleme bei Datenbankabfragen](performance-probleme-bei-datenbankabfragen.md)
<br/>  Nicht indexfreundliche Abfragemuster schaffen Performance-Engpässe bei Datenbankoperationen.
- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Während Tabellen wachsen, verschlechtern sich Abfragen, die Indizes nicht nutzen können, progressiv, weil vollständige Scans länger dauern.

## Causes ▼

- [Lücken in der Kompetenzentwicklung](luecken-in-der-kompetenzentwicklung.md)
<br/>  Entwicklern fehlt Wissen darüber, wie Datenbankabfrageoptimierer funktionieren und welche Muster die Indexnutzung verhindern.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Code-Reviews, die Abfrageperformance nicht bewerten, verpassen Muster, die die Indexnutzung verhindern.
- [Ineffiziente Datenbankindizierung](ineffiziente-datenbankindizierung.md)
<br/>  Schlecht designte Indizes passen möglicherweise nicht zu Abfragemustern, was den Effekt indexunfreundlicher Abfragen verstärkt.

## Detection Methods ○

- **Analyse des Abfrageausführungsplans:** Dies ist die primäre Methode. Nutzen Sie immer `EXPLAIN` oder `EXPLAIN ANALYZE`, um zu verstehen, wie die Datenbank Ihre Abfragen ausführt. Suchen Sie nach `Seq Scan` oder „Full Table Scan" bei großen Tabellen, bei denen ein Index erwartet wird.
- **Datenbank-Slow-Query-Logs:** Konfigurieren Sie Ihre Datenbank, um langsame Abfragen zu protokollieren, und überprüfen Sie diese Logs regelmäßig.
- **Automatisierte Abfrageperformance-Werkzeuge:** Viele APM-Werkzeuge oder Datenbank-Monitoring-Lösungen können ineffiziente Abfragen identifizieren und Verbesserungen vorschlagen.
- **Code-Review:** Entwickler sollten sich häufiger Muster bewusst sein, die die Indexnutzung während Code-Reviews verhindern.

## Examples
Ein Nutzer-Suchfeature fragt eine `users`-Tabelle mit `WHERE LOWER(email) = 'john.doe@example.com'` ab. Obwohl `email` indiziert ist, verhindert die `LOWER()`-Funktion, dass der Index genutzt wird, was zu einem vollständigen Tabellenscan führt. Das Umschreiben zu `WHERE email ILIKE 'john.doe@example.com'` (falls nicht-case-sensitive Suche benötigt und von der Datenbank unterstützt wird) oder die Sicherstellung, dass die Anwendung Groß-/Kleinschreibung vor der Abfrage handhabt, kann dies beheben. In einem anderen Fall nutzt eine Report-Abfrage `WHERE product_code LIKE '%ABC%'`. Ein Index auf `product_code` existiert, aber das führende Wildcard verhindert seine Nutzung. Wenn das Suchmuster immer ein Suffix ist, könnte ein umgekehrter Index genutzt werden, oder die Abfrage falls möglich umgeschrieben werden. Dieses Problem hebt die Wichtigkeit hervor, zu verstehen, wie Datenbankoptimierer funktionieren, und Abfragen zu schreiben, die es ihnen erlauben, bestehende Indizes effektiv zu nutzen. Es ist eine häufige Quelle von Performance-Engpässen, besonders in Anwendungen mit komplexen Reporting- oder Suchfunktionen.
