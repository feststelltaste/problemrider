---
title: Datenbankoptimierung
description: Anpassung von Datenbankdesign und -konfiguration für optimale
  Performance.
category:
- Database
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/database-optimization/
problems:
- database-query-performance-issues
- n-plus-one-query-problem
- slow-database-queries
- slow-response-times-for-lists
- high-number-of-database-queries
- inefficient-database-indexing
- incorrect-index-type
- unused-indexes
- index-fragmentation
- queries-that-prevent-index-usage
- long-running-database-transactions
- lock-contention
- high-database-resource-utilization
- database-connection-leaks
- misconfigured-connection-pools
- incorrect-max-connection-pool-size
- deadlock-conditions
- graphql-complexity-issues
- imperative-data-fetching-logic
- lazy-loading
- long-running-transactions
layout: solution
lang: de
en_slug: query-optimization-process
related_solutions:
- slug: materialized-views
  similarity: 0.8
- slug: index-lifecycle-management
  similarity: 0.8
- slug: efficient-algorithms
  similarity: 0.8
- slug: resource-usage-optimization
  similarity: 0.75
- slug: caching-strategy
  similarity: 0.75
- slug: denormalization
  similarity: 0.75
---

## Description

Datenbankoptimierung stimmt Indizes, Abfragepläne und Serverkonfiguration auf das tatsächliche aktuelle Datenvolumen und die Zugriffsmuster einer Datenbank ab und gewinnt Performance zurück, die ein Legacy-Schema verlor, während es weit über den Maßstab hinauswuchs, für den es ursprünglich entworfen war. Da Legacy-Tabellen häufig nur mit einem Primärschlüsselindex erstellt wurden und veraltete Statistiken sich still über Jahre ansammeln, kann eine Abfrage, die einst Millisekunden brauchte, am Ende einen vollständigen Tabellenscan über Millionen Zeilen durchführen, ohne dass jemand die Abfrage selbst angefasst hat — die Ursache liegt vollständig im eigenen Zustand der Datenbank, nicht in der Anwendung. Die Aktivierung von Slow-Query-Logging, um die echte Arbeitslast zu finden, und dann das Hinzufügen von Indizes und die Auffrischung von Statistiken, um sie ihr anzupassen, liefert routinemäßig 100-fache Verbesserungen, ohne eine Zeile Anwendungscode anzufassen, obwohl schreibintensive Legacy-Tabellen ebenso leicht durch Überindizierung geschädigt werden können, die ohne Rücksicht auf Einfügungskosten hinzugefügt wurde.

## How to Apply ◆

> Legacy-Datenbanken wurden für ursprüngliche Datenvolumina und Zugriffsmuster entworfen, die sich oft dramatisch von der aktuellen Produktionslast unterscheiden — systematische Abfrageoptimierung gewinnt Performance zurück, ohne teure Hardware-Upgrades oder riskante Schema-Neuschreibungen zu erfordern.

- Aktivieren Sie Slow-Query-Logging als ersten Schritt (`log_min_duration_statement` in PostgreSQL, `slow_query_log` in MySQL) und lassen Sie es mindestens einen vollständigen Geschäftszyklus laufen, um die echte Abfrage-Arbeitslast zu erfassen, nicht nur, was Entwickler für langsam halten.
- Priorisieren Sie Optimierung nach Auswirkung: eine Abfrage, die 10.000 Mal pro Tag läuft und 300 ms braucht, ist wichtiger als eine, die einmal läuft und 10 Sekunden braucht — berechnen Sie die gesamte pro Abfrage verbrauchte Zeit, nicht nur die Worst-Case-Dauer.
- Verwenden Sie EXPLAIN (ANALYZE) für jede Kandidatenabfrage, bevor Sie Änderungen vornehmen; Legacy-Datenbanken haben häufig veraltete Tabellenstatistiken, die den Query Planner dazu bringen, einen sequenziellen Scan auf einer Tabelle mit mehreren Millionen Zeilen zu wählen, wenn ein Indexscan Millisekunden dauern würde — das vorherige Ausführen von ANALYZE kann schlechte Abfragepläne sofort ohne jegliche Schemaänderungen beheben.
- Fügen Sie Indizes hinzu, die zu den tatsächlichen WHERE-, JOIN- und ORDER-BY-Mustern der teuersten Abfragen passen; in Legacy-Systemen wurden viele Tabellen nur mit einem Primärschlüsselindex erstellt und verlassen sich für jedes andere Zugriffsmuster auf vollständige Tabellenscans.
- Identifizieren und schreiben Sie N+1-Abfragemuster neu, die von ORMs oder handgeschriebenen Schleifen im Anwendungscode generiert werden; diese gehören zu den häufigsten Performance-Killern in Legacy-Java-EE- und älteren Rails-Anwendungen, in denen Lazy Loading die unhinterfragte Standardeinstellung war.
- Ersetzen Sie `SELECT *` durch benannte Spaltenlisten in der gesamten Codebasis; Legacy-Systeme rufen oft ganze breite Zeilen ab, wenn nur zwei oder drei Spalten benötigt werden, was I/O, Speicher und Netzwerkbandbreite verschwendet.
- Stimmen Sie Datenbankserver-Speichereinstellungen auf die tatsächlich verfügbare Hardware ab; Legacy-Datenbanken laufen häufig mit Standard-Buffer-Pool-Größen (128 MB) auf Servern mit 64 GB RAM, was den Großteil des Maschinenspeichers ungenutzt lässt, während die Datenbank die Festplatte hämmert.
- Trennen Sie lesehäufige Reporting-Abfragen auf Read Replicas, sodass sie nicht mit transaktionalen Abfragen um dieselben Datenbankressourcen konkurrieren — eine Änderung, die Legacy-Architekturen fast nie enthalten, die aber beide Arbeitslasten dramatisch verbessert.

## Tradeoffs ⇄

> Datenbankabfrageoptimierung in Legacy-Systemen liefert einige der größten verfügbaren Performance-Gewinne beim niedrigsten strukturellen Risiko, erfordert aber laufende Aufmerksamkeit, während sich Datenvolumina und Abfragemuster weiterentwickeln.

**Vorteile:**

- Gut platzierte Indizes können einzelne Abfragezeiten von Sekunden auf Millisekunden reduzieren — Verbesserungen von 100-fach oder mehr —, ohne Anwendungslogik anzufassen oder Datenintegrität zu riskieren.
- Optimierung verschiebt die Notwendigkeit teurer Datenbankmigrationen, Sharding oder Hardware-Upgrades, indem bestehende Infrastruktur weit effizienter gemacht wird, was die produktive Lebensdauer von Legacy-Systemen verlängert.
- Slow-Query-Analyse offenbart, welche Teile der Anwendung die meiste Datenbanklast erzeugen, und lenkt Refactoring-Aufwand dorthin, wo er die höchste Rendite hat, statt dorthin, wo Entwickler Probleme vermuten.
- Die Behebung von N+1-Mustern und der Ersatz von SELECT * erfordert oft nur Änderungen in der Datenzugriffsschicht und lässt Geschäftslogik unangetastet — risikoarme Verbesserungen in Codebasen, in denen das Anfassen von Geschäftslogik als gefährlich gilt.
- Reduzierte Lock Contention durch schnellere Abfragen verbessert den gesamten Systemdurchsatz unter gleichzeitiger Last und adressiert die Beschwerde „funktioniert gut mit 20 Nutzern, bricht mit 200 Nutzern", die in alternden Webanwendungen üblich ist.

**Kosten und Risiken:**

- Indizes verlangsamen Schreiboperationen und verbrauchen Festplattenspeicher; Überindizierung von Legacy-Tabellen mit hohen Einfügeraten — üblich in Logging- oder Event-Sourcing-Mustern — kann Schreibvorgänge dramatisch verlangsamen.
- Abfrageoptimierung für aktuelle Zugriffsmuster passt möglicherweise nicht zu künftigen Mustern; ein für die heute häufigste Abfrage gebauter Covering-Index wird zu totem Gewicht, wenn die Anwendung ändert, wie sie Daten abruft.
- Konfigurationstuning (Buffer Pools, Work Memory, Checkpoint-Intervalle) erfordert Verständnis von Datenbankinternas, das in Teams, die ein Legacy-System ohne seine ursprünglichen Architekten geerbt haben, selten vorhanden ist.
- Die Partitionierung großer Legacy-Tabellen zur Verbesserung der Abfrageperformance führt zu Cross-Partition-Abfragekomplexität und erfordert sorgfältige Koordination mit allen Fremdschlüsselbeziehungen und Anwendungsabfragen, die die Partitionsgrenze überspannen.
- Optimierungsaufwand kann zu einem Ersatz für die Adressierung der zugrunde liegenden architektonischen Probleme werden (z. B. gemeinsame Datenbanken, fehlende Caching-Schichten), die exzessive Datenbanklast überhaupt erst verursachen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen Abfrageoptimierung, angewendet in realistischen Legacy-System-Kontexten.

Ein mittelgroßes E-Commerce-Unternehmen hatte eine Legacy-PHP-Anwendung, gestützt auf eine MySQL-Datenbank, die über ein Jahrzehnt von 100.000 auf 40 Millionen Produktdatensätze gewachsen war. Die Produktsuchseite hatte sich von einer 200-ms-Antwort auf einen 8-Sekunden-Timeout unter normaler Last verschlechtert. Die Aktivierung des Slow-Query-Logs offenbarte, dass die Suchabfrage bei jeder Anfrage einen vollständigen Tabellenscan auf der Produkttabelle durchführte, weil die WHERE-Klausel Spalten in einer Reihenfolge kombinierte, die zu keinem bestehenden Index passte. Das Hinzufügen eines zusammengesetzten Index auf `(category_id, status, price)`, passend zum Filtermuster der Abfrage, reduzierte die Suchabfrage auf 18 ms, ohne eine einzige Zeile Anwendungscode zu ändern.

Eine Regierungsbehörde führte nächtliche Batch-Jobs gegen eine Oracle-Datenbank aus, die zwölf Jahre lang gewachsen war und nun 800 Millionen Zeilen in ihrer primären Transaktionstabelle hielt. Die Jobs, geschrieben in den frühen 2000er-Jahren, nutzten `SELECT *`, um vollständige Zeilen abzurufen, bevor im Anwendungscode gefiltert wurde, und brauchten 14 Stunden zum Abschluss — was in die Geschäftszeiten überlief. Eine Überprüfung der Ausführungspläne zeigte, dass das Neuschreiben dreier Abfragen zur Auswahl nur der benötigten Spalten und das Hinzufügen zweier partieller Indizes auf die Teilmenge aktiver Datensätze die Batch-Laufzeit auf unter 3 Stunden reduzierte, was das nächtliche Fenster einhielt, ohne jegliche Änderungen an der Batch-Logikstruktur.

Das Legacy-Schadensmanagementsystem eines Versicherungsunternehmens erlebte periodische Verlangsamungen, die mit hohen gleichzeitigen Nutzerzahlen korrelierten. Die Untersuchung mit EXPLAIN ANALYZE offenbarte, dass eine kritische Abfrage, die vier Tabellen verband, eine geschätzte Zeilenzahl von 200 hatte, während die tatsächliche Zahl während Spitzenzeiten 2 Millionen betrug, was den Planner dazu brachte, eine Nested-Loop-Join-Strategie zu wählen, die im kleinen Maßstab funktionierte, aber unter Last zusammenbrach. Das Ausführen von ANALYZE zur Auffrischung der Statistiken und die Anpassung von `default_statistics_target` für die schief verteilten Spalten brachte den Planner dazu, zu einem Hash-Join zu wechseln, was die Abfragezeit von 45 Sekunden auf 400 ms senkte und die periodischen Verlangsamungen vollständig beseitigte.
