---
title: Graphdatenbanken
description: Ermöglichung der Speicherung und Abfrage verknüpfter Daten in Form
  von Knoten und Kanten.
category:
- Database
- Performance
problems:
- slow-database-queries
- database-query-performance-issues
- algorithmic-complexity-problems
- database-schema-design-problems
- complex-domain-model
layout: solution
lang: de
en_slug: graph-databases
related_solutions:
- slug: nosql-databases
  similarity: 0.85
- slug: denormalization
  similarity: 0.75
- slug: materialized-views
  similarity: 0.75
- slug: data-replication
  similarity: 0.75
- slug: object-relational-mapping-orm
  similarity: 0.7
- slug: data-partitioning
  similarity: 0.7
---

## Description

Eine Graphdatenbank speichert Daten als Knoten, die Entitäten repräsentieren, und Kanten, die die Beziehungen zwischen ihnen repräsentieren, wobei beide Eigenschaften tragen können, und sie wird mit traversierungsorientierten Sprachen wie Cypher oder Gremlin abgefragt statt mit SQL-Joins. Ihr prägendes Performance-Merkmal ist, dass das Durchqueren einer Beziehung ungefähr gleich viel kostet, unabhängig davon, wie groß der Gesamtdatensatz wächst, weil eine Abfrage nur den lokalen Kanten folgt, die mit relevanten Knoten verbunden sind, statt ganze Tabellen zu durchsuchen oder zu verbinden. Legacy-relationale Systeme modellieren häufig von Natur aus graphförmige Domänen — Organisationshierarchien, Berechtigungsvererbung, Empfehlungsnetzwerke, Abhängigkeitsketten — mittels Fremdschlüsseln und Join-Tabellen, was rekursive oder mehrstufige Abfragen erzwingt, die stark verkommen, während Verschachtelungstiefe oder Datenvolumen zunehmen, und oft zu einer der hartnäckigeren Quellen langsamer Datenbankabfragen in einem alternden Schema werden. Nur die graphförmige Teilmenge des Datenmodells in eine dedizierte Graphdatenbank zu migrieren, während tabellarische und aggregationslastige Daten im bestehenden relationalen Speicher verbleiben, zielt auf diese spezifische Problemklasse ab, ohne eine vollständige Datenbankmigration zu erfordern. Der Zielkonflikt ist architektonisch: Das Legacy-System gewinnt eine zweite Datenbanktechnologie und die betriebliche und Synchronisationskomplexität, die damit einhergeht, zwei Datenspeicher konsistent zu halten, im Austausch für Traversierungsabfragen, die von einer mehrsekündigen, tiefenbegrenzten Belastung zu einer millisekundenschnellen, tiefenunabhängigen Fähigkeit werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Datenmodelle im Legacy-System, die von Natur aus graphförmig sind: soziale Netzwerke, Organisationshierarchien, Abhängigkeitsbäume, Empfehlungs-Engines
- Bewerten Sie, ob aktuelle Performance-Probleme von tiefen JOIN-Ketten oder rekursiven Abfragen herrühren, die von Graph-Traversierung profitieren würden
- Wählen Sie eine für Maßstab und Abfragemuster geeignete Graphdatenbank (Neo4j, Amazon Neptune, JanusGraph)
- Modellieren Sie die Domäne als Knoten (Entitäten) und Kanten (Beziehungen) mit Eigenschaften auf beiden
- Migrieren Sie die graphförmige Teilmenge der Daten in die Graphdatenbank, während Nicht-Graph-Daten im relationalen Speicher verbleiben
- Implementieren Sie Synchronisation zwischen relationaler und Graphdatenbank, falls beide konsistent bleiben müssen
- Nutzen Sie Graph-Abfragesprachen (Cypher, Gremlin), um Beziehungstraversierungen auszudrücken, die in SQL umständlich sind

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Exzellent im Durchqueren von Beziehungen, was für tiefe Graph-Abfragen um Größenordnungen schneller sein kann als SQL-JOINs
- Natürliche Modellierung verknüpfter Daten ohne komplexe Join-Tabellen
- Die Abfrageperformance bleibt stabil, während Daten wachsen, weil Traversierung von der lokalen Graphstruktur abhängt, nicht von der Gesamtdatensatzgröße
- Ermöglicht die Entdeckung von Mustern und Pfaden, die in relationalen Datenbanken unpraktikabel abzufragen sind

**Kosten und Risiken:**
- Fügt dem Stack eine weitere Datenbanktechnologie hinzu, was die betriebliche Komplexität erhöht
- Graphdatenbanken sind weniger ausgereift und haben kleinere Ökosysteme als relationale Datenbanken
- Nicht für alle Workloads geeignet: tabellarische Daten und Aggregationen werden von relationalen Datenbanken besser gehandhabt
- Das Team muss neue Abfragesprachen und Datenmodellierungsparadigmen lernen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Zugriffskontrollsystem speicherte Organisationshierarchien und Berechtigungsvererbung in einer relationalen Datenbank. Die Bestimmung der effektiven Berechtigungen eines Nutzers erforderte rekursive Abfragen über mehrere Ebenen von Gruppenmitgliedschaft und Rollenvererbung, was für tief verschachtelte Organisationen über 10 Sekunden dauerte. Das Team migrierte das Berechtigungsmodell zu Neo4j, wo eine Cypher-Abfrage den gesamten Berechtigungsgraphen unabhängig von der Tiefe in Millisekunden durchqueren konnte. Die relationale Datenbank blieb die maßgebliche Quelle für Nutzer- und Gruppendaten, wobei Änderungen über Events zu Neo4j synchronisiert wurden. Berechtigungsprüfungen sanken von Sekunden auf einstellige Millisekunden, und das System konnte nun Organisationen mit beliebig tiefen Hierarchien unterstützen.
