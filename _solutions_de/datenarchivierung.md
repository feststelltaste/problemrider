---
title: Datenarchivierung
description: Auslagerung selten benötigter Daten auf kosteneffizientere Speichermedien.
category:
- Database
- Performance
problems:
- unbounded-data-growth
- gradual-performance-degradation
- slow-database-queries
- high-database-resource-utilization
- database-schema-design-problems
- unbounded-data-structures
- inadequate-test-data-management
- retention-obligations-block-change
layout: solution
lang: de
en_slug: data-archiving
related_solutions:
- slug: data-partitioning
  similarity: 0.8
- slug: data-replication
  similarity: 0.8
- slug: redundant-data-storage
  similarity: 0.8
- slug: materialized-views
  similarity: 0.75
- slug: platform-independent-data-storage
  similarity: 0.75
- slug: compression
  similarity: 0.75
---

## Description

Datenarchivierung verschiebt Daten, die nicht mehr aktiv benötigt werden — typischerweise identifiziert nach Alter oder abnehmender Zugriffshäufigkeit — aus der primären, performance-kritischen Speicherebene in günstigeren, langsameren Speicher, wo sie weiterhin verfügbar bleiben, aber das Tagesgeschäft nicht mehr belasten. Anders als Löschung bewahrt Archivierung die Daten für Compliance, Audit oder gelegentliche historische Abfrage, verlagert sie aber, sodass der aktive Datensatz, gegen den die Anwendung abfragt, klein und schnell bleibt. Diese Unterscheidung ist in Legacy-Systemen von großer Bedeutung, wo Aufbewahrungspflichten oder schlichte institutionelle Vorsicht Jahre oder Jahrzehnte an Transaktionshistorie in denselben Tabellen belassen haben, die den täglichen Betrieb antreiben, was dazu führt, dass Indizes aufblähen, Backups länger dauern und selbst routinemäßige Abfragen sich verlangsamen, während sich die Datenbank-Engine durch Daten arbeitet, die niemand mehr tatsächlich nutzt. Ein gut gestalteter Archivierungsprozess ist automatisiert und umkehrbar: Er läuft nach einem definierten Zeitplan gegen klare Kriterien, und er wird mit einem Wiederherstellungspfad gepaart, sodass archivierte Datensätze noch produziert werden können, wenn ein Audit oder eine Kundenanfrage sie erfordert. Weil Anwendungsabfragen in Legacy-Systemen oft ohne jede Datumsbegrenzungsannahme geschrieben wurden, erfordert die Einführung von Archivierung typischerweise auch die Aktualisierung dieser Abfragen, um explizit auf den aktiven Datensatz abzuzielen, was eine Lücke schließt, die unbegrenztes Wachstum überhaupt erst unbemerkt anhäufen ließ.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Analysieren Sie Datenzugriffsmuster, um Daten zu identifizieren, die nach einem bestimmten Alter selten abgefragt werden
- Definieren Sie Archivierungsrichtlinien basierend auf Geschäftsanforderungen: regulatorische Aufbewahrungsfristen, Auditbedürfnisse und Zugriffshäufigkeit
- Implementieren Sie automatisierte Archivierungsprozesse, die Daten nach Zeitplan von Hot Storage zu Cold Storage verschieben
- Stellen Sie sicher, dass archivierte Daten für Compliance und Ad-hoc-Abfragen zugänglich bleiben, auch wenn Zugriffszeiten langsamer sind
- Testen Sie die Archivierungs- und Wiederherstellungsprozesse regelmäßig, um zu verifizieren, dass archivierte Daten bei Bedarf wiederhergestellt werden können
- Aktualisieren Sie Anwendungsabfragen, um nach Datumsbereichen zu filtern, sodass sie natürlich auf dem aktiven Datensatz operieren
- Koordinieren Sie mit Stakeholdern, um zu definieren, was für jede Domäne „aktive" versus „archivierte" Daten ausmacht

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Reduziert die aktive Datensatzgröße, was Abfrageperformance und Backup-Zeiten verbessert
- Senkt Speicherkosten, indem selten genutzte Daten auf günstigere Medien verschoben werden
- Vereinfacht Datenbankwartungsaufgaben wie Index-Neuaufbau und Schema-Migrationen
- Verbessert die Anwendungsperformance, indem Working Sets handhabbar gehalten werden

**Kosten und Risiken:**
- Archivierte Daten sind langsamer zugänglich, was Nutzer frustrieren kann, die historische Informationen brauchen
- Archivierungsprozesse fügen operative Komplexität hinzu und erfordern Überwachung
- Unsachgemäße Archivierung kann regulatorische Aufbewahrungsanforderungen verletzen
- Anwendungslogik könnte Aktualisierungen benötigen, um sowohl aktive als auch archivierte Daten transparent abzufragen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Schadenbearbeitungssystem eines Versicherungsunternehmens hatte 12 Jahre an Schadensdaten in einer einzigen Datenbank angehäuft, insgesamt über 500 Millionen Datensätze. Die Abfrageperformance hatte sich so weit verschlechtert, dass selbst einfache Abfragen mehrere Sekunden dauerten. Das Team implementierte eine Datenarchivierungsstrategie, die Schadensfälle älter als drei Jahre in eine separate Archivdatenbank auf günstigerem Speicher verschob. Die aktive Datenbank schrumpfte um 75 Prozent, und die Abfrageperformance kehrte zu Sub-Sekunden-Werten zurück. Für regulatorische Audits, die historische Daten erforderten, griff eine dedizierte Abfrageschnittstelle auf das Archiv zu, mit akzeptablen Antwortzeiten von wenigen Sekunden pro Abfrage.
