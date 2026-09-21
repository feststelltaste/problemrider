---
title: Lazy Loading
description: Die Nutzung von Lazy Loading in einem ORM-Framework führt zu einer großen
  Anzahl unnötiger Datenbankabfragen, was die Anwendungsperformance erheblich verschlechtern
  kann.
category:
- Code
- Database
- Performance
related_problems:
- slug: slow-database-queries
  similarity: 0.75
- slug: n-plus-one-query-problem
  similarity: 0.75
- slug: imperative-data-fetching-logic
  similarity: 0.7
- slug: high-number-of-database-queries
  similarity: 0.7
- slug: poor-caching-strategy
  similarity: 0.65
- slug: slow-application-performance
  similarity: 0.65
solutions:
- caching-strategy
- efficient-algorithms
- lazy-evaluation
- query-optimization-process
- profiling
- denormalization
- materialized-views
- pagination
layout: problem
lang: de
en_slug: lazy-loading
---

## Description
Lazy Loading ist ein Designmuster, das genutzt wird, um die Initialisierung eines Objekts zu verzögern, bis es tatsächlich benötigt wird. Dies kann in manchen Fällen ein nützliches Muster sein, kann aber auch zu Performance-Problemen führen. Im Kontext eines Object-Relational-Mapping(ORM)-Frameworks kann Lazy Loading zum N+1-Abfrageproblem führen. Dies liegt daran, dass das ORM eine separate Abfrage für jedes Objekt ausführt, das lazy geladen wird. Dies kann zu einer großen Anzahl unnötiger Datenbankabfragen führen, was die Anwendungsperformance erheblich verschlechtern kann.

## Indicators ⟡
- Die Anwendung macht eine große Anzahl kleiner, schneller Abfragen statt einiger weniger größerer, langsamerer Abfragen.
- Die Anwendung ist langsam, obwohl der Datenbankserver nicht unter hoher Last steht.
- Die Datenbank-Logs sind voll von ähnlich aussehenden Abfragen.
- Die Anwendung nutzt ein ORM-Framework, und man ist sich nicht sicher, ob es korrekt konfiguriert ist.

## Symptoms ▲

- [N+1-Abfrageproblem](n-plus-1-abfrageproblem.md)
<br/>  Lazy Loading verursacht direkt das N+1-Abfragemuster, bei dem jede lazy geladene Beziehung eine separate Datenbankabfrage auslöst.
- [Hohe Anzahl an Datenbankabfragen](hohe-anzahl-an-datenbankabfragen.md)
<br/>  Jede lazy geladene Assoziation erzeugt zusätzliche Abfragen, was die Gesamtanzahl der Datenbankaufrufe pro Anfrage vervielfacht.
- [Langsame Datenbankabfragen](langsame-datenbankabfragen.md)
<br/>  Der kumulative Effekt vieler lazy geladener Abfragen verschlechtert die Gesamtdatenbankperformance und Antwortzeiten.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Die übermäßige Anzahl an Datenbank-Roundtrips durch Lazy Loading lässt die Anwendung für Nutzer träge erscheinen.
- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Die Flut kleiner Abfragen durch Lazy Loading verbraucht übermäßig Datenbank-CPU- und Verbindungsressourcen.

## Causes ▼

- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit ORM-Verhalten nicht vertraut sind, nutzen möglicherweise Standard-Lazy-Loading-Einstellungen, ohne die Performance-Implikationen zu verstehen.
- [Bequemlichkeitsgetriebene Entwicklung](bequemlichkeitsgetriebene-entwicklung.md)
<br/>  Lazy Loading ist oft die Standard- und bequemste ORM-Option, und Entwickler wählen sie, ohne die Performance-Kompromisse zu bewerten.

## Detection Methods ○
- **Application Performance Monitoring (APM):** APM-Werkzeuge können oft das N+1-Abfrageproblem erkennen und markieren, das ein verbreitetes Symptom von Lazy Loading ist.
- **SQL-Logging:** Aktivierung des SQL-Loggings in der Anwendung oder Datenbank und Untersuchung der Logs auf eine große Anzahl ähnlich aussehender Abfragen.
- **Code-Review:** Während Code-Reviews gezielte Suche nach Code, der Lazy Loading nutzt.
- **ORM-Profiling:** Manche ORM-Frameworks bieten Werkzeuge zum Profiling der Performance von Abfragen.

## Examples
Eine Webanwendung nutzt ein ORM-Framework, um Daten aus der Datenbank abzurufen. Die Anwendung hat eine Seite, die eine Liste von Nutzern und deren Beiträgen anzeigt. Die Anwendung nutzt Lazy Loading, um die Beiträge für jeden Nutzer abzurufen. Das bedeutet, dass die Anwendung zunächst eine Abfrage ausführt, um die Liste der Nutzer zu erhalten. Dann führt sie für jeden Nutzer eine weitere Abfrage aus, um deren Beiträge zu erhalten. Dies resultiert in einer großen Anzahl unnötiger Datenbankabfragen, was die Seite langsam lädt. Das Problem könnte gelöst werden, indem Eager Loading genutzt wird, um die Nutzer und ihre Beiträge in einer einzigen Abfrage abzurufen.
