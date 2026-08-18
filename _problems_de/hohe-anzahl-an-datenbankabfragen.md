---
title: Hohe Anzahl an Datenbankabfragen
description: Eine einzelne Nutzeranfrage löst eine unerwartet große Anzahl von Datenbankabfragen
  aus, was zu Performance-Verschlechterung und erhöhter Datenbanklast führt.
category:
- Database
- Performance
related_problems:
- slug: n-plus-one-query-problem
  similarity: 0.8
- slug: slow-database-queries
  similarity: 0.8
- slug: database-query-performance-issues
  similarity: 0.8
- slug: high-connection-count
  similarity: 0.75
- slug: imperative-data-fetching-logic
  similarity: 0.75
- slug: high-database-resource-utilization
  similarity: 0.7
solutions:
- query-optimization-process
- api-calls-optimization
- batch-processing
- denormalization
- materialized-views
- pagination
- index-lifecycle-management
- profiling
- performance-measurements
- continuous-performance-monitoring
- typed-schema-extraction
layout: problem
lang: de
en_slug: high-number-of-database-queries
---

## Description
Eine hohe Anzahl an Datenbankabfragen ist ein verbreitetes Performance-Problem in datenbankgetriebenen Anwendungen. Es tritt auf, wenn eine einzelne Nutzeranfrage eine unerwartet große Anzahl von Datenbankabfragen auslöst. Dies kann aus verschiedenen Gründen geschehen, etwa dem N+1-Abfrageproblem, fehlendem Caching oder einer schlecht gestalteten Datenzugriffsschicht. Eine hohe Anzahl an Datenbankabfragen kann zu einer Reihe von Problemen führen, darunter langsame Anwendungsperformance, hohe Datenbank-Ressourcennutzung und ein schlechtes Nutzererlebnis.

## Indicators ⟡
- Die Anwendung ist langsam, obwohl der Datenbankserver nicht unter hoher Last steht.
- Die Datenbank-Logs sind voll von ähnlich aussehenden Abfragen.
- Die Anwendung macht viele kleine, schnelle Abfragen statt einiger weniger größerer, langsamerer Abfragen.
- Die Anwendung nutzt keine Caching-Schicht.

## Symptoms ▲

- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Eine große Anzahl an Abfragen pro Anfrage erhöht die CPU-, Speicher- und I/O-Last auf dem Datenbankserver.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Die kumulative Latenz vieler Datenbank-Roundtrips pro Anfrage verlangsamt direkt die Antwortzeiten der Anwendung.
- [Hohe API-Latenz](hohe-api-latenz.md)
<br/>  API-Endpunkte, die übermäßige Datenbankabfragen auslösen, erleben erhöhte Antwortzeiten aufgrund des akkumulierten Abfrage-Overheads.

## Causes ▼

- [N+1-Abfrageproblem](n-plus-1-abfrageproblem.md)
<br/>  Das einzelne Laden verwandter Entitäten in einer Schleife erzeugt eine zusätzliche Abfrage für jedes Element, was die Gesamtabfragenanzahl vervielfacht.
- [Imperative Datenabruflogik](imperative-datenabruflogik.md)
<br/>  Das Abrufen von Daten in Schleifen statt der Nutzung von Batch- oder deklarativen Ansätzen erzeugt übermäßig viele einzelne Abfragen.
- [Lazy Loading](lazy-loading.md)
<br/>  Lazy-geladene Beziehungen lösen zusätzliche Abfragen aus, wenn auf sie zugegriffen wird, was oft unerwartet die Abfragenanzahl vervielfacht.
- [Schlechte Caching-Strategie](schlechte-caching-strategie.md)
<br/>  Ohne Caching werden dieselben Daten wiederholt aus der Datenbank abgerufen, statt aus dem Speicher bedient zu werden.

## Detection Methods ○
- **Application Performance Monitoring (APM):** APM-Werkzeuge können oft eine hohe Anzahl an Datenbankabfragen erkennen und markieren.
- **SQL-Logging:** Aktivierung des SQL-Loggings in der Anwendung oder Datenbank und Untersuchung der Logs auf eine große Anzahl von Abfragen, die in kurzer Zeit ausgeführt werden.
- **Code-Review:** Während Code-Reviews gezielte Suche nach Code, der eine große Anzahl an Datenbankabfragen macht.
- **Lasttests:** Nutzung von Lasttests, um zu sehen, wie sich die Anwendung unter hoher Last verhält.

## Examples
Eine Webanwendung hat eine Seite, die eine Liste von Produkten anzeigt. Für jedes Produkt zeigt sie auch den Namen der Kategorie an, zu der das Produkt gehört. Die Anwendung führt zunächst eine Abfrage aus, um die Liste der Produkte zu erhalten. Dann führt sie für jedes Produkt eine weitere Abfrage aus, um den Namen der Kategorie zu erhalten. Dies resultiert in einer großen Anzahl an Datenbankabfragen, was die Seite langsam lädt. Das Problem könnte durch die Nutzung einer einzigen Abfrage gelöst werden, die die Produkt- und Kategorietabellen verknüpft.
