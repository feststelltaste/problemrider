---
title: Evolutionäres Datenbankdesign
description: Schrittweise Weiterentwicklung von Datenbankschemata durch versionskontrollierte
  Migrationen.
category:
- Database
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/evolutionary-database-design/
problems:
- data-migration-complexities
- data-migration-integrity-issues
- database-schema-design-problems
- schema-evolution-paralysis
- shared-database
- silent-data-corruption
- cross-system-data-synchronization-problems
- unbounded-data-growth
- long-running-database-transactions
- entity-attribute-value-overuse
layout: solution
lang: de
en_slug: evolutionary-database-design
related_solutions:
- slug: backward-compatible-schema-migrations
  similarity: 0.75
- slug: query-optimization-process
  similarity: 0.75
- slug: ci-cd-pipeline
  similarity: 0.75
- slug: containerized-databases
  similarity: 0.7
- slug: strangler-fig-pattern
  similarity: 0.7
- slug: nosql-databases
  similarity: 0.7
---

## Description

Evolutionäres Datenbankdesign verwaltet Schemaänderungen als versionskontrollierte Migrationsskripte, ausgeführt durch ein Werkzeug wie Flyway oder Liquibase, statt als manuelle ALTER-Anweisungen, direkt gegen Produktion angewandt von wem auch immer gerade Bereitschaftsdienst hat. Legacy-Datenbanken werden typischerweise genau als diese Art statischen, manuell veränderten Artefakts verwaltet, mit Schemahistorie, die nirgendwo existiert außer im Gedächtnis dessen, der jede Änderung vorgenommen hat — was das lokale Reproduzieren von Produktion oder das Nachdenken darüber, was sich geändert hat und warum, so schwierig macht. Eine Baseline-Migration für das aktuelle Schema zu etablieren und das Expand-and-Contract-Muster für alles anzuwenden, was einen laufenden Konsumenten brechen würde, bringt Schemaevolution unter dieselbe Review- und CI-Disziplin wie Anwendungscode, auf Kosten echter zusätzlicher Komplexität während der Übergangsperiode, die jede solche Änderung erfordert.

## How to Apply ◆

> Legacy-Datenbanken werden typischerweise als statische, manuell veränderte Artefakte verwaltet — die Einführung versionskontrollierter Migrationsskripte verwandelt sie in erstklassige, pflegbare Komponenten des Systems.

- Etablieren Sie ein Migrationswerkzeug (Flyway, Liquibase oder Alembic je nach Stack) als ersten Schritt, noch bevor Sie das Schema anfassen; konfigurieren Sie es, das aktuelle Produktionsschema als Baseline „Version Null" zu erkennen, sodass alle zukünftigen Änderungen ab diesem Punkt verfolgt werden.
- Wenden Sie nie wieder Schemaänderungen direkt in einem Datenbankclient an; jede Änderung, egal wie klein, muss durch ein Migrationsskript laufen, das in der Versionskontrolle neben dem Anwendungscode lebt, der davon abhängt.
- Wenden Sie das Expand-and-Contract-Muster für jede Änderung an, die laufende Konsumenten brechen würde: Fügen Sie zuerst die neue Spalte oder Tabelle hinzu, migrieren Sie Code und Daten, sie zu nutzen, und löschen Sie die alte Struktur erst, nachdem alle Abhängigen umgestellt haben — dies ist besonders wichtig in Legacy-Systemen, wo mehrere Anwendungen eine einzige Datenbank teilen.
- Testen Sie Migrationen gegen eine wiederhergestellte Kopie des Produktions-Backups in einer Staging-Umgebung, bevor Sie sie auf Produktion anwenden; Legacy-Tabellen mit Millionen von Zeilen verhalten sich sehr anders als kleine Entwicklungsdatensätze.
- Führen Sie für große Datenmigrationen auf alternden Tabellen die Datentransformation als separaten Hintergrund-Batch-Prozess aus statt inline im Migrationsskript — Inline-Migrationen können Tabellen stundenlang sperren und verlängerte Ausfälle verursachen.
- Schreiben Sie für jede Migration Skripte auf, auch für solche, die unumkehrbar erscheinen; die Übung, Rollback durchzudenken, offenbart Annahmen und Risiken, die sonst nur während eines Vorfalls auftauchen würden.
- Nutzen Sie Migrationsskripte, um angehäufte Schemaschulden schrittweise zu bereinigen: Fügen Sie fehlende Indizes hinzu, setzen Sie Beschränkungen durch, die zuvor nur im Code geprüft wurden, benennen Sie irreführende Spalten um — dieselbe Expand-and-Contract-Disziplin gilt.
- Verhindern Sie, dass Entwickler bereits angewandte Migrationen bearbeiten; setzen Sie dies durch Code-Review-Richtlinien durch und, wo das Tooling es erlaubt, durch Checksummenvalidierung, die Modifikationen an historischen Migrationsdateien erkennt.

## Tradeoffs ⇄

> Evolutionäres Datenbankdesign bringt Schemaänderungen unter dieselben Qualitätskontrollen wie Anwendungscode, aber die erforderliche Disziplin ist höher als bei zustandslosem Code, und die Konsequenzen von Fehlern sind schwerer umzukehren.

**Vorteile:**

- Jede Schemaänderung ist überprüfbar, auditierbar und reproduzierbar — ein Entwickler kann das Repository klonen und alle Migrationen ausführen, um ein zu Produktion identisches Schema zu erhalten, was die „Referenzdatenbank" eliminiert, die nur eine Person einzurichten weiß.
- Schemaänderungen laufen durch dieselbe CI/CD-Pipeline wie Anwendungscode, was kontinuierliche Auslieferung von Features ermöglicht, die beide Schichten überspannen, ohne den separaten, hochzeremoniellen DBA-Genehmigungsprozess, der in Legacy-Organisationen üblich ist.
- Das Expand-and-Contract-Muster erlaubt Zero-Downtime-Schemaänderungen an Systemen, die zuvor Wartungsfenster selbst für triviale Spaltenzusätze erforderten.
- Historische Migrationen liefern eine archäologische Aufzeichnung, die neuen Teammitgliedern hilft zu verstehen, warum sich das Schema so entwickelt hat, wie es hat — unschätzbar in Legacy-Systemen, wo institutionelles Gedächtnis verloren gegangen ist.
- Angehäufte Schemaschulden können schrittweise abgebaut werden, ohne das Alles-oder-nichts-Risiko eines Big-Bang-Redesigns, das Legacy-Modernisierungsprojekte oft entgleisen lässt.

**Kosten und Risiken:**

- Das Expand-and-Contract-Muster vervielfacht die Anzahl der für komplexe Änderungen nötigen Migrationen und führt eine Periode von Synchronisationslogik ein, die während des Übergangs operative Komplexität hinzufügt.
- Fehler in Migrationen auf Legacy-Datenbanken mit großen Tabellen und ohne angemessene Testumgebung können mehrstündige Ausfälle verursachen — Testen gegen produktionsgroße Daten ist teuer, aber essenziell.
- Rollback destruktiver Migrationen (das Löschen einer Spalte, die sich als noch benötigt herausstellt) erfordert kompensierende Migrationen oder Datenbankwiederherstellungen, die in großen Legacy-Datenbanken langsam und schmerzhaft sind.
- Teams, die an Ad-hoc-SQL-Skripte gewöhnt sind, müssen die Disziplin verinnerlichen, nie angewandte Migrationen zu bearbeiten — Verstöße verursachen Schemadivergenz über Umgebungen hinweg, die schwer zu diagnostizieren ist.
- Bestehende Legacy-Datenbanken haben oft undokumentierte manuelle Änderungen, Trigger und gespeicherte Prozeduren, die in keiner Migrationshistorie erscheinen; die anfängliche Baseline-Migration muss durch Inspektion konstruiert werden, was zeitaufwendig und fehleranfällig ist.

## How It Could Be

> Die folgenden Szenarien veranschaulichen den Wert und die Herausforderungen der Einführung evolutionären Datenbankdesigns in Legacy-System-Kontexten.

Ein Versicherungsunternehmen, das ein fünfzehn Jahre altes Schadenbearbeitungssystem betrieb, musste der zentralen Schadenstabelle ein neues Feld hinzufügen, um eine regulatorische Berichtsanforderung zu unterstützen. Zuvor hätte ihr DBA eine ALTER-TABLE-Anweisung direkt in Produktion während eines Samstags-Wartungsfensters ausgeführt. Nach der Einführung von Flyway durchlief das Migrationsskript Code-Review, wurde auf eine aus einem Produktions-Backup wiederhergestellte Staging-Umgebung angewandt — was enthüllte, dass die Tabelle auf 80 Millionen Zeilen angewachsen war und die Spaltenzugabe zwölf Minuten dauern würde — und wurde dann während eines geplanten Fensters mit der Anwendung im Nur-Lesen-Modus auf Produktion angewandt. Die Migration lief erfolgreich, und das Team hatte eine dauerhafte Aufzeichnung dessen, was sich genau geändert hatte und wann.

Ein Logistikunternehmen entdeckte, dass drei separate Java-Services und zwei Legacy-Perl-Batch-Jobs alle dieselbe PostgreSQL-Datenbank teilten. Als ein Team eine Spalte von `shipment_ref` zu `reference_number` umbenennen musste, um Konsistenz herzustellen, drohte die Änderung, alle fünf Konsumenten zu brechen. Mittels Expand-and-Contract fügte das Team die neue Spalte hinzu, fügte einen Datenbank-Trigger hinzu, um beide Spalten synchron zu halten, aktualisierte jede Anwendung eine nach der anderen über zwei Wochen und entfernte dann die alte Spalte und den Trigger in einer finalen Bereinigungsmigration. Die gesamte Änderung geschah ohne Ausfallzeit und ohne Koordinationsfenster — etwas, das das Team zuvor für unmöglich gehalten hatte.

Ein Gesundheitstechnologieunternehmen, das eine Legacy-Oracle-Datenbank modernisieren wollte, erkannte, dass Jahre manueller ALTER-TABLE-Anweisungen verschiedener DBAs das Produktionsschema und das Entwicklungsschema auf subtile Weise auseinandergebracht hatten. Kein Entwickler konnte die Produktionsumgebung lokal reproduzieren. Durch das Ausführen von Liquibase im „Off"-Modus gegen Produktion erzeugten sie ein initiales Changelog, das den aktuellen Zustand repräsentierte, committeten es als Baseline und verlangten ab diesem Punkt, dass alle Änderungen durch Migrationsskripte laufen. Innerhalb von sechs Monaten konnten neue Entwickler eine lokale, mit Produktion übereinstimmende Umgebung in unter zehn Minuten einrichten — eine Aufgabe, die zuvor das Klonen eines sorgfältig gepflegten Entwickler-Snapshots erfordert hatte, der selbst oft Monate veraltet war.
