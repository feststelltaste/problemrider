---
title: Index-Lifecycle-Management
description: Behandlung von Datenbank-Indizes als gepflegte Assets — überprüft
  anhand tatsächlicher Abfragemuster, auf Nutzung gemessen und entfernt, wenn sie
  ihre Kosten nicht mehr rechtfertigen.
category:
- Database
- Performance
- Operations
problems:
- inefficient-database-indexing
- unused-indexes
- incorrect-index-type
- index-fragmentation
- queries-that-prevent-index-usage
- high-number-of-database-queries
- slow-database-queries
- n-plus-one-query-problem
- long-running-database-transactions
- slow-application-performance
- imperative-data-fetching-logic
- lock-contention
- long-running-transactions
- poor-caching-strategy
- entity-attribute-value-overuse
layout: solution
lang: de
en_slug: index-lifecycle-management
related_solutions:
- slug: query-optimization-process
  similarity: 0.8
- slug: materialized-views
  similarity: 0.65
- slug: evolutionary-database-design
  similarity: 0.6
- slug: efficient-algorithms
  similarity: 0.6
- slug: resource-usage-optimization
  similarity: 0.6
- slug: typed-schema-extraction
  similarity: 0.6
---

## Description

Index-Lifecycle-Management ist die Praxis, Indizes als Assets mit laufenden Kosten zu behandeln statt als einmalige Ergänzungen: periodisch anhand der tatsächlich ausgeführten Abfragen überprüft, auf Nutzung gemessen, gegen Fragmentierung gepflegt und entfernt, wenn sie ihre Kosten nicht mehr rechtfertigen. Indizes sammeln sich in Legacy-Datenbanken auf charakteristische Weise an. Jeder wurde hinzugefügt, um eine spezifische langsame Abfrage zu beheben, oft Jahre zuvor, von jemandem, der inzwischen gegangen ist, und keiner wurde je entfernt. Das Ergebnis ist eine Tabelle mit vierzehn Indizes, mehrere redundant, ein paar seit einer Abfrage-Umschreibung 2016 ungenutzt, und jeder Schreibvorgang zahlt die Kosten, alle davon zu pflegen. Währenddessen sind die heute langsamen Abfragen langsam, weil ihr Zugriffsmuster nie indiziert wurde. Das Problem ist kein Mangel an Indizierungsfähigkeit, sondern die Abwesenheit eines Prozesses, der Entscheidungen erneut aufgreift.

## How to Apply ◆

> Der Index-Bestand einer Legacy-Datenbank ist eine historische Aufzeichnung vergangener Performance-Vorfälle, kein Design — weshalb er üblicherweise gleichzeitig über- und unterindiziert ist.

- **Inventarisieren Sie die aktuellen Indizes** mit ihrer Größe, und beziehen Sie Nutzungsstatistiken von der Datenbank — jede gängige Engine verfolgt, wie oft jeder Index vom Planer genutzt wird. Indizes mit null Lesezugriffen über einen vollen Geschäftszyklus sind reiner Schreib-Overhead.
- **Beginnen Sie bei den Abfragen, nicht den Tabellen.** Erfassen Sie die tatsächliche Arbeitslast aus dem Slow-Query-Log oder der Statement-Statistik-View, gerankt nach insgesamt verbrauchter Zeit statt nach Einzeldauer. Eine Abfrage, die 40 Millisekunden braucht und zwei Millionen Mal täglich läuft, zählt mehr als eine, die nachts acht Sekunden braucht.
- **Suchen Sie nach Redundanz**: Ein Index auf `(a)` ist redundant, wenn `(a, b)` existiert. Zusammengesetzte-Index-Präfixe sind die häufigste Quelle unnötiger Indizes in alten Schemata, weil jeder von jemandem hinzugefügt wurde, der nicht prüfte, was bereits existierte.
- **Prüfen Sie, ob die Spaltenreihenfolge zu den Abfragemustern passt.** Ein zusammengesetzter Index ist nur für Abfragen nutzbar, die seine führenden Spalten einschränken. Ein Index, dessen Reihenfolge für eine nicht mehr existierende Abfrage gewählt wurde, ist oft der Grund, warum eine aktuelle Abfrage ihn nicht nutzen kann.
- **Identifizieren Sie Abfragen, die ihre Indizes zunichtemachen** — eine auf die indizierte Spalte angewendete Funktion, eine Typinkonsistenz, die Konvertierung erzwingt, ein führender Platzhalter, ein `OR` über Spalten hinweg. Diese brauchen meist eine geänderte Abfrage statt eines weiteren Index, und stattdessen einen hinzuzufügen ist, wie Index-Bestände wachsen.
- **Wählen Sie den Index-Typ bewusst**, wo die Engine mehrere anbietet. Partielle Indizes für Abfragen, die immer nach derselben Bedingung filtern, und abdeckende Indizes für heiße Lesepfade übertreffen häufig das Hinzufügen eines weiteren einfachen Index und kosten weniger in der Pflege.
- **Planen Sie Wartung** für Fragmentierung und Statistikaktualität gemäß den Anforderungen der Engine. Veraltete Statistiken lassen den Planer schlechte Entscheidungen treffen, selbst wenn die Indizes korrekt sind, und dies ist eine häufige Ursache für eine Abfrage, die letzten Monat schnell war und jetzt langsam ist.
- **Entfernen Sie ungenutzte Indizes auf reversible Weise**: Machen Sie sie für den Planer unsichtbar, wenn die Engine dies unterstützt, oder entfernen Sie sie in einem Fenster, in dem Neuerstellung machbar ist, und überwachen Sie vor dem Abschluss. Einen Index zu entfernen, der nur von einem Quartalsbericht genutzt wird, ist ein Fehler, der ein Quartal später entdeckt wird.
- **Überprüfen Sie in einem Takt** — vierteljährlich oder nach jeder bedeutenden Änderung der Abfragemuster — und erfassen Sie den Grund für jeden Index. Ein Index, dessen Zweck dokumentiert ist, kann später bewertet werden; einer ohne Grund wird nie entfernt werden.
- **Verifizieren Sie gegen realistisches Datenvolumen.** Index-Verhalten hängt von Kardinalität und Verteilung ab, sodass eine gegen einen kleinen Testdatensatz validierte Entscheidung fast nichts über die Produktion aussagt.

## Tradeoffs ⇄

> Gepflegte Indizes beschleunigen Lesevorgänge und verringern Schreib-Overhead, aber jede Änderung birgt Risiko an einem Live-System, und die Überprüfung verbraucht Fachzeit.

**Vorteile:**

- Die Leselegistung verbessert sich, wo es zählt, weil die Indizes von der aktuellen Arbeitslast statt von historischen Vorfällen abgeleitet sind.
- Schreibperformance verbessert sich, und der Speicherbedarf sinkt, wenn redundante und ungenutzte Indizes entfernt werden — oft ein erheblicher Effekt bei Tabellen mit vielen Indizes.
- Langsame Abfragen, die kein Index beheben wird, werden als Abfrageprobleme identifiziert, was die korrekte Diagnose ist und weitere Index-Anhäufung verhindert.
- Backup-, Wiederherstellungs- und Migrationszeiten verkürzen sich, was direkt für Wartungsfenster und Disaster Recovery zählt.
- Die dokumentierte Begründung macht künftige Überprüfung möglich und durchbricht den Zyklus, in dem Indizes nur je hinzugefügt werden.

**Kosten und Risiken:**

- Das Entfernen eines selten, aber kritisch genutzten Index verschlechtert diesen Pfad erheblich, und die Entdeckung kann ein Quartal entfernt sein.
- Index-Änderungen an großen Live-Tabellen können teuer oder sperrintensiv sein, abhängig von der Engine, und erfordern Wartungsfenster, die schwer zu bekommen sind.
- Nutzungsstatistiken setzen sich bei manchen Engines beim Neustart zurück und spiegeln nur den Beobachtungszeitraum wider, sodass ein kurzes Fenster irreführende Schlüsse produziert.
- Die Arbeit erfordert Datenbankexpertise, die viele Teams, die Legacy-Systeme pflegen, nicht mehr im Haus haben.
- Das Hinzufügen von Indizes zur Behebung von Lesevorgängen verlagert Kosten auf Schreibvorgänge, und bei schreiblastigen Tabellen kann der Nettoeffekt negativ sein, auf Weisen, die erst bei steigender Last offensichtlich werden.

## How It Could Be

Ein Team, das eine Auftragsverwaltungsdatenbank pflegte, untersuchte, warum die nächtliche Batch-Verarbeitung über zwei Jahre von 90 Minuten auf über vier Stunden gewachsen war. Die naheliegenden Verdächtigen waren Datenvolumen und Abfragepläne. Ein Index-Inventar fand 142 Indizes über die 30 größten Tabellen, von denen Nutzungsstatistiken über ein volles Quartal zeigten, dass 31 nie gelesen worden waren und weitere 18 redundante Präfixe zusammengesetzter Indizes waren. Allein die Auftragspositionstabelle trug 11 Indizes, was jedem Insert etwa 40 Prozent Overhead hinzufügte — und der Batch-Job fügte nachts mehrere Millionen Zeilen ein. Das Entfernen der 31 ungenutzten und 18 redundanten Indizes, in zwei Phasen mit Überwachung dazwischen, brachte das Batch-Fenster auf 105 Minuten zurück. Keine Abfrage wurde langsamer.

Dieselbe Überprüfung veränderte, wie das Team eine chronisch langsame Kundensuche handhabte. Ihre drei vorherigen Versuche hatten jeweils einen Index hinzugefügt, keiner davon half. Die Untersuchung der Abfrage zeigte, dass sie `UPPER()` auf die Nachnamensspalte anwandte, was jeden Index auf dieser Spalte unbrauchbar machte — der Planer hatte unabhängig davon, wie viele Indizes existierten, einen vollständigen Scan durchgeführt. Das Hinzufügen eines Case-insensitive-Ausdrucksindex brachte die Abfrage von 3,2 Sekunden auf 15 Millisekunden und erlaubte, zwei der drei Indizes zu entfernen, die in vorherigen Versuchen hinzugefügt worden waren. Das Team übernahm daraus eine Regel: Kein Index wird zur Behebung einer langsamen Abfrage hinzugefügt, bevor nicht jemand den Ausführungsplan gelesen und bestätigt hat, dass die Abfrage einen nutzen kann.
