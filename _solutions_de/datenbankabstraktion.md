---
title: Datenbankabstraktion
description: Umsetzung von Datenbankzugriffen über eine abstrahierte Schicht.
category:
- Database
- Architecture
problems:
- technology-lock-in
- vendor-lock-in
- tight-coupling-issues
- data-migration-complexities
- database-schema-design-problems
- difficult-to-test-code
- incorrect-index-type
layout: solution
lang: de
en_slug: database-abstraction
related_solutions:
- slug: object-relational-mapping-orm
  similarity: 0.9
- slug: abstraction-layers
  similarity: 0.85
- slug: abstracted-file-system-access
  similarity: 0.85
- slug: platform-independent-data-storage
  similarity: 0.85
- slug: protocol-abstraction
  similarity: 0.8
- slug: abstraction
  similarity: 0.8
---

## Description

Datenbankabstraktion fügt eine dedizierte Datenzugriffsschicht — einen ORM, Repository-Schnittstellen oder eine handgebaute Adapterschicht — zwischen Geschäftslogik und dem rohen SQL oder den datenbankspezifischen Konstrukten ein, von denen ein Legacy-System abhängt, sodass Konsumenten von Daten mit einer Abstraktion interagieren statt direkt mit dem Dialekt und den Features der zugrunde liegenden Datenbank-Engine. Abfragen und Persistenzlogik durchlaufen diese Schicht, statt inline über die gesamte Codebasis geschrieben zu werden, und jede Operation, die echt herstellerspezifische Funktionalität erfordert, wird in klar markierte Adapter-Module isoliert, statt verstreut im Geschäftscode zu bleiben. Dies ist zentral für Legacy-Modernisierung, weil Legacy-Codebasen über die Jahre häufig Tausende roher, herstellerspezifischer SQL-Anweisungen anhäufen — proprietäre Funktionen, dialektspezifische Syntax, eingebettete Stored-Procedure-Aufrufe —, die die gesamte Anwendung eng an einen Datenbankhersteller koppeln und es effektiv unmöglich machen, Geschäftslogik ohne eine Live-Datenbankverbindung zu testen. Einmal abstrahiert, machen dieselben Schnittstellen, die den Datenbankhersteller verstecken, es auch unkompliziert, eine In-Memory- oder Test-Implementierung für Unit-Tests einzusetzen, was Korrektheitstests von der Datenbankverfügbarkeit entkoppelt. Die Abfragen eines Legacy-Systems in die Abstraktionsschicht zu migrieren ist angesichts des typischerweise beteiligten schieren Volumens notwendigerweise schrittweise, aber es verwandelt eine Datenbankherstellermigration von einer Neuschreibung der gesamten Anwendung in eine begrenzte Übung, die sich auf die Abstraktionsschicht und ihre Adapter konzentriert.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Führen Sie einen ORM oder eine Datenzugriffsschicht (z. B. Hibernate, Entity Framework, SQLAlchemy) zwischen Geschäftslogik und rohem SQL ein
- Kapseln Sie allen Datenbankzugriff hinter Repository-Schnittstellen, die die zugrunde liegende Datenbanktechnologie verstecken
- Ersetzen Sie datenbankspezifische SQL-Syntax (Stored Procedures, proprietäre Funktionen) durch portable Äquivalente, wo möglich
- Isolieren Sie unvermeidlich datenbankspezifische Operationen in klar markierte Adapter-Module
- Nutzen Sie Datenbankmigrationswerkzeuge, die portables DDL erzeugen, statt handgeschriebener datenbankspezifischer Skripte
- Implementieren Sie das Repository-Muster mit In-Memory-Implementierungen für Unit-Tests
- Migrieren Sie schrittweise rohe SQL-Abfragen zur Abstraktionsschicht, priorisieren Sie die am häufigsten modifizierten Codepfade

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht Migration zwischen Datenbankherstellern, ohne Geschäftslogik neu zu schreiben
- Verbessert die Testbarkeit, indem Datenbanklogik mit In-Memory-Implementierungen getestet werden kann
- Zentralisiert Abfrageoptimierung und Caching-Belange in einer Schicht
- Reduziert die Verbreitung von SQL über die Codebasis, was die Pflegbarkeit verbessert

**Kosten und Risiken:**
- ORM-Abstraktionen können ineffiziente Abfragen erzeugen, die schlechter performen als handgeschriebenes SQL
- Komplexe Legacy-Abfragen, die herstellerspezifische Features nutzen, lassen sich möglicherweise nicht sauber auf die Abstraktion abbilden
- Die Abstraktionsschicht selbst führt eine Lernkurve und potenzielle Fehler ein
- Performance-kritische Operationen müssen möglicherweise die Abstraktion umgehen, was Inkonsistenz schafft
- Die Migration einer großen Legacy-Codebasis mit Tausenden roher SQL-Anweisungen ist ein mehrjähriger Aufwand

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Java-Anwendung enthielt über 2.000 Oracle-spezifische SQL-Abfragen, verstreut über ihre Codebasis, einschließlich PL/SQL-Stored-Procedure-Aufrufen und Oracle-spezifischer Datumsfunktionen. Als das Unternehmen entschied, zu PostgreSQL zu migrieren, um Lizenzkosten zu reduzieren, musste jede Abfrage modifiziert werden. Das Team führte JPA-Repositories ein und migrierte Abfragen über 18 Monate schrittweise zu JPQL. Sie isolierten die 50 Abfragen, die echt datenbankspezifische Features benötigten, in Adapter-Klassen mit sowohl Oracle- als auch PostgreSQL-Implementierungen. Dieser Ansatz erlaubte ihnen, während der Migration beide Datenbanken parallel laufen zu lassen, wobei die Adapter-Auswahl über Konfiguration gesteuert wurde, und schloss die Migration letztlich ohne jede Geschäftslogikänderung ab.
