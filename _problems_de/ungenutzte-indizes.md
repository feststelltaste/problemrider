---
title: Ungenutzte Indizes
description: Die Datenbank hat Indizes, die nie von Abfragen genutzt werden, aber
  dennoch Speicherplatz verbrauchen und Overhead zu Schreiboperationen hinzufügen.
category:
- Code
- Performance
related_problems:
- slug: queries-that-prevent-index-usage
  similarity: 0.7
- slug: index-fragmentation
  similarity: 0.7
- slug: inefficient-database-indexing
  similarity: 0.65
- slug: incorrect-index-type
  similarity: 0.6
- slug: database-query-performance-issues
  similarity: 0.6
- slug: high-number-of-database-queries
  similarity: 0.55
solutions:
- query-optimization-process
- performance-measurements
- monitoring
- regular-maintenance-and-updates
- static-code-analysis
- continuous-performance-monitoring
- index-lifecycle-management
- data-modeling
- load-testing
- capacity-planning
layout: problem
lang: de
en_slug: unused-indexes
---

## Description
Ungenutzte Indizes sind Datenbankindizes, die von keiner Abfrage genutzt werden. Während Indizes entscheidend für die Beschleunigung des Datenabrufs sind, verbrauchen ungenutzte weiterhin Festplattenspeicher und, noch wichtiger, fügen Overhead zu Schreiboperationen (INSERT, UPDATE, DELETE) hinzu, weil die Datenbank den Index bei jeder Änderung der zugrunde liegenden Daten aktualisieren muss. Die regelmäßige Identifikation und Entfernung ungenutzter Indizes ist ein Schlüsselaspekt der Datenbankwartung und Performance-Optimierung, da sie Speicher zurückgewinnt und unnötige Verarbeitung reduziert.

## Indicators ⟡
- Die Datenbank nutzt viel Festplattenspeicher, obwohl der Datensatz klein ist.
- Schreiboperationen sind langsam, obwohl die Datenbank nicht stark ausgelastet ist.
- Backups und Wiederherstellungen brauchen lange.
- Die Datenbank nutzt viel Speicher, selbst wenn sie nicht stark ausgelastet ist.

## Symptoms ▲

- [Ressourcenverschwendung](ressourcenverschwendung.md)
<br/>  Ungenutzte Indizes verbrauchen Festplattenspeicher, Speicher und CPU-Zyklen während Schreiboperationen, ohne irgendeinen Abfragevorteil zu bieten.
- [Langsame Datenbankabfragen](langsame-datenbankabfragen.md)
<br/>  Der Overhead der Wartung ungenutzter Indizes verlangsamt die gesamten Datenbankoperationen, besonders schreiblastige Workloads.

## Causes ▼

- [Probleme im Datenbankschema-Design](probleme-im-datenbankschema-design.md)
<br/>  Ohne regelmäßige Datenbankwartung und -auditierung häufen sich ungenutzte Indizes über die Zeit an, während sich Abfragen ändern, aber Indizes bestehen bleiben.
- [Monitoring-Lücken](monitoring-luecken.md)
<br/>  Ohne Monitoring der Indexnutzungsstatistiken haben Teams keine Sichtbarkeit darüber, welche Indizes tatsächlich genutzt werden.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Schlechte Dokumentation darüber, warum Indizes erstellt wurden, macht es schwierig festzustellen, ob sie noch benötigt werden.

## Detection Methods ○

- **Datenbank-Monitoring-Werkzeuge:** Die meisten modernen Datenbanksysteme liefern Statistiken zur Indexnutzung (z. B. `pg_stat_user_indexes` in PostgreSQL, `sys.dm_db_index_usage_stats` in SQL Server, `information_schema.statistics` in MySQL kombiniert mit Abfrageprotokollen).
- **Analyse von Abfrageausführungsplänen:** Regelmäßige Analyse von `EXPLAIN`-Plänen für übliche Abfragen zur Sicherstellung, dass sie die effizientesten Indizes nutzen.
- **Automatisierte Index-Berater:** Manche Datenbankmanagementsysteme oder Drittanbieter-Werkzeuge bieten automatisierte Indexempfehlungen und Nutzungsanalyse.
- **Periodische Datenbank-Audits:** Planung regelmäßiger Überprüfungen von Datenbankschema und Indexnutzung.

## Examples
Eine Legacy-E-Commerce-Anwendung hat eine `orders`-Tabelle mit einem Index auf `customer_id`, der vor Jahren für eine spezifische Berichtsabfrage erstellt wurde. Dieser Bericht wurde seitdem eingestellt, aber der Index bleibt bestehen, was Overhead zu jeder neuen platzierten Bestellung hinzufügt, ohne irgendeinen Abfrage-Performance-Vorteil zu bieten. In einem anderen Fall erstellt ein Entwickler einen Index auf `(spalte_A, spalte_B)`, erstellt aber später einen weiteren Index auf `(spalte_A)`. Der Query-Optimizer der Datenbank bevorzugt möglicherweise den kleineren `(spalte_A)`-Index für Abfragen, die nur `spalte_A` betreffen, was den zusammengesetzten Index für diese Abfragen teilweise ungenutzt macht. Ungenutzte Indizes sind eine Form technischer Schulden im Datenbankmanagement. Während sie keine direkten Anwendungsfehler verursachen, verschlechtern sie still die Schreib-Performance, verschwenden Ressourcen und erschweren die Datenbankwartung, besonders in großskaligen Systemen.
