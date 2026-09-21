---
title: Imperative Datenabruflogik
description: Der Anwendungscode ist so geschrieben, dass er Daten in einer Schleife
  abruft, statt einen effizienteren, deklarativen Ansatz zu nutzen, was zu Performance-Problemen
  führt.
category:
- Architecture
- Database
- Performance
related_problems:
- slug: n-plus-one-query-problem
  similarity: 0.8
- slug: high-number-of-database-queries
  similarity: 0.75
- slug: slow-database-queries
  similarity: 0.7
- slug: lazy-loading
  similarity: 0.7
- slug: inefficient-code
  similarity: 0.7
- slug: poor-caching-strategy
  similarity: 0.7
solutions:
- efficient-algorithms
- reactive-programming
- query-optimization-process
- caching-strategy
- object-relational-mapping-orm
- cqrs
- materialized-views
- profiling
- performance-measurements
- index-lifecycle-management
- typed-schema-extraction
layout: problem
lang: de
en_slug: imperative-data-fetching-logic
---

## Description
Imperative Datenabruflogik ist ein verbreitetes Performance-Problem in datenbankgetriebenen Anwendungen. Es tritt auf, wenn der Anwendungscode so geschrieben ist, dass er Daten in einer Schleife abruft, statt einen effizienteren, deklarativen Ansatz zu nutzen. Dies kann zu einer Reihe von Problemen führen, darunter das N+1-Abfrageproblem, langsame Anwendungsperformance und eine hohe Anzahl an Datenbankabfragen. Imperative Datenabruflogik ist oft ein Zeichen für fehlende Erfahrung mit deklarativer Programmierung oder eine fehlende klare Datenabrufstrategie.

## Indicators ⟡
- Der Anwendungscode enthält Schleifen, die Daten aus der Datenbank abrufen.
- Die Anwendung macht eine große Anzahl kleiner, schneller Abfragen statt einiger weniger größerer, langsamerer Abfragen.
- Die Anwendung ist langsam, obwohl der Datenbankserver nicht unter hoher Last steht.
- Die Datenbank-Logs sind voll von ähnlich aussehenden Abfragen.

## Symptoms ▲

- [Hohe Anzahl an Datenbankabfragen](hohe-anzahl-an-datenbankabfragen.md)
<br/>  Das Abrufen von Daten in Schleifen erzeugt einzelne Abfragen für jede Iteration, was zu einer übermäßigen Anzahl an Datenbankaufrufen führt.
- [N+1-Abfrageproblem](n-plus-1-abfrageproblem.md)
<br/>  Imperative Abrufmuster, die verwandte Daten einzeln laden, sind die direkte Implementierungsursache von N+1-Abfrageproblemen.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Die kumulierte Latenz vieler sequenzieller Datenbank-Roundtrips verschlechtert die Antwortzeiten der Anwendung erheblich.
- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Das übermäßige Abfragevolumen durch imperatives Abrufen erhöht die CPU- und Speichernutzung auf dem Datenbankserver.

## Causes ▼

- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit deklarativen Datenzugriffsmustern und ORM-Best-Practices nicht vertraut sind, greifen standardmäßig auf imperatives, schleifenbasiertes Abrufen zurück.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Ohne Code-Review bleiben ineffiziente Datenabrufmuster unentdeckt und etablieren sich in der Codebasis.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ohne etablierte Datenzugriffsmuster und -standards implementieren Entwickler Abruflogik inkonsistent und ineffizient.

## Detection Methods ○
- **Code-Review:** Während Code-Reviews gezielte Suche nach Schleifen, die Datenbankabfragen enthalten.
- **Application Performance Monitoring (APM):** APM-Werkzeuge können oft das N+1-Abfrageproblem erkennen und markieren, das ein verbreitetes Symptom imperativer Datenabruflogik ist.
- **SQL-Logging:** Aktivierung des SQL-Loggings in der Anwendung oder Datenbank und Untersuchung der Logs auf eine große Anzahl ähnlich aussehender Abfragen.
- **Statische Analyse:** Nutzung statischer Analysewerkzeuge zur Identifikation von Schleifen, die Datenbankabfragen enthalten.

## Examples
Eine Webanwendung hat eine Seite, die eine Liste von Produkten und deren Preise anzeigt. Die Anwendung führt zunächst eine Abfrage aus, um die Liste der Produkte zu erhalten. Dann führt sie für jedes Produkt eine weitere Abfrage aus, um den Preis zu erhalten. Dies ist ein Beispiel für imperative Datenabruflogik. Das Problem könnte durch die Nutzung einer einzigen Abfrage gelöst werden, die die Produkt- und Preistabellen verknüpft. Dies wäre eine effizientere und deklarativere Art, die Daten abzurufen.
