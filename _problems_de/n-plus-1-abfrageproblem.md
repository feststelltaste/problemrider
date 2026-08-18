---
title: N+1-Abfrageproblem
description: Eine Anwendung führt zahlreiche unnötige Datenbankaufrufe aus, um zusammengehörige
  Daten abzurufen, wo eine einzige, effizientere Abfrage ausreichen würde, was erhebliche
  Performance-Verschlechterung verursacht.
category:
- Database
- Performance
related_problems:
- slug: high-number-of-database-queries
  similarity: 0.8
- slug: imperative-data-fetching-logic
  similarity: 0.8
- slug: slow-database-queries
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.75
- slug: slow-response-times-for-lists
  similarity: 0.7
- slug: database-query-performance-issues
  similarity: 0.7
solutions:
- efficient-algorithms
- api-calls-optimization
- denormalization
- object-relational-mapping-orm
- query-optimization-process
- profiling
- performance-measurements
- code-reviews
- static-code-analysis
- continuous-performance-monitoring
- index-lifecycle-management
- typed-schema-extraction
layout: problem
lang: de
en_slug: n-plus-one-query-problem
---

## Description
Das N+1-Abfrageproblem ist ein häufiges Performance-Problem, das auftritt, wenn eine Anwendung eine Abfrage ausführt, um eine Liste von Elementen abzurufen, und dann für jedes dieser Elemente eine zusätzliche Abfrage ausführt, um zusammengehörige Daten abzurufen. Dies resultiert in einer großen Anzahl kleiner, ineffizienter Abfragen, was die Anwendung erheblich verlangsamen kann. Dieses Problem wird oft durch Object-Relational-Mapping-Frameworks (ORM) eingeführt, wenn sie nicht sorgfältig genutzt werden. Die Lösung des N+1-Problems beinhaltet typischerweise das Abrufen aller benötigten Daten in einer einzigen, effizienteren Abfrage.

## Indicators ⟡
- Eine Seite, die eine Liste von Elementen anzeigt, lädt langsam.
- Die Datenbank ist stark belastet, obwohl die Anwendung nicht viel Arbeit verrichtet.
- Sie sehen eine große Anzahl ähnlich aussehender Abfragen in Ihren Datenbank-Logs.
- Ihre Anwendung führt viele kleine, schnelle Abfragen aus statt weniger größerer, langsamerer Abfragen.

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Die exzessive Anzahl an Datenbank-Roundtrips verschlechtert direkt die Anwendungsantwortzeiten, besonders auf Seiten, die Listen zusammengehöriger Daten anzeigen.
- [Hohe Anzahl an Datenbankabfragen](hohe-anzahl-an-datenbankabfragen.md)
<br/>  Das N+1-Muster erzeugt ein großes Volumen einzelner Abfragen, wo weniger, optimierte Abfragen ausreichen würden.
- [Performance-Probleme bei Datenbankabfragen](performance-probleme-bei-datenbankabfragen.md)
<br/>  Die Flut einzelner Abfragen aus N+1-Mustern überlastet die Datenbank, was die Gesamtabfrageperformance verschlechtert.
- [Negatives Nutzerfeedback](negatives-nutzerfeedback.md)
<br/>  Nutzer erleben langsame Seitenladezeiten und träge Oberflächen als direkte Folge exzessiver Datenbankabfragen.

## Causes ▼

- [Lazy Loading](lazy-loading.md)
<br/>  ORM-Lazy-Loading löst transparent einzelne Abfragen für jeden Zugriff auf verwandte Objekte aus, was es leicht macht, das N+1-Muster unwissentlich einzuführen.
- [Imperative Datenabruflogik](imperative-datenabruflogik.md)
<br/>  Das Schreiben von Datenabruf in Schleifen statt deklarativer Batch-Abfragen erzeugt natürlicherweise das N+1-Muster.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Ohne Reviews, die Datenbankzugriffsmuster genau prüfen, schleichen sich N+1-Abfragen unbemerkt in die Produktion ein.
- [Wissenslücken](wissensluecken.md)
<br/>  Entwickler, die mit ORM-Verhalten oder Datenbankoptimierungstechniken nicht vertraut sind, führen unbeabsichtigt N+1-Muster ein.

## Detection Methods ○

- **Application Performance Monitoring (APM):** APM-Werkzeuge können N+1-Abfragemuster oft automatisch erkennen und markieren.
- **SQL-Logging:** Aktivierung von SQL-Logging in Ihrer Anwendung oder Datenbank und Untersuchung der Logs auf eine große Anzahl identischer Abfragen, die in kurzer Zeit ausgeführt werden.
- **Code-Review:** Während Code-Reviews gezielt nach Schleifen suchen, die Datenbankabfragen oder Aufrufe von Datenabruffunktionen enthalten.
- **Spezialisierte Bibliotheken:** Manche Bibliotheken und Werkzeuge (wie `bullet` für Ruby on Rails) sind speziell dafür konzipiert, das N+1-Abfrage-Antipattern während der Entwicklung zu erkennen.

## Examples
Eine Blog-Anwendung zeigt eine Liste der 10 neuesten Beiträge auf ihrer Startseite an. Für jeden Beitrag zeigt sie auch den Namen des Autors an. Der Code führt zunächst eine Abfrage aus, um die 10 Beiträge zu erhalten (`SELECT * FROM posts ORDER BY created_at DESC LIMIT 10`). Dann durchläuft er diese 10 Beiträge und führt für jeden eine neue Abfrage aus, um den Namen des Autors zu erhalten (`SELECT name FROM authors WHERE id = ?`). Dies resultiert in 1 (für die Beiträge) + 10 (für die Autoren) = 11 Abfragen insgesamt.

**Problematischer Code**

Dieses Codestück zeigt das problematische N+1-Muster:

```python
posts = Post.objects.all().limit(10)
for post in posts:
  # Diese Zeile löst eine neue Abfrage für jeden einzelnen Beitrag aus!
  print(f"{post.title} by {post.author.name}")
```

**Korrigierte Version**
Die korrigierte Version nutzt Eager Loading. Dies wird alle Beiträge und ihre Autoren in nur zwei Abfragen abrufen (oder einer mit Join):

```python
posts = Post.objects.select_related('author').all().limit(10)
for post in posts:
  print(f"{post.title} by {post.author.name}")
```

Dies ist ein extrem häufiges Problem in Anwendungen, die ORMs nutzen, besonders in Legacy-Codebasen, wo Performance während der ursprünglichen Entwicklung kein primäres Anliegen war.
