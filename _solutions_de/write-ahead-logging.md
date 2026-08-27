---
title: Write-Ahead Logging
description: Aufzeichnung von Änderungen in einem dauerhaften
  Append-Only-Log, bevor sie angewendet werden.
category:
- Database
- Architecture
problems:
- silent-data-corruption
- data-migration-integrity-issues
- cascade-failures
- system-outages
- debugging-difficulties
- missing-rollback-strategy
- long-running-database-transactions
- inconsistent-behavior
layout: solution
lang: de
en_slug: write-ahead-logging
related_solutions:
- slug: transactions
  similarity: 0.75
- slug: backup-and-recovery
  similarity: 0.7
- slug: logging
  similarity: 0.7
- slug: regular-backups
  similarity: 0.7
- slug: audit-trail-management
  similarity: 0.7
- slug: timestamping
  similarity: 0.7
---

## Description

Write-Ahead Logging ist eine Haltbarkeitstechnik, bei der jede beabsichtigte Änderung an einem Datenspeicher zuerst in einem sequenziellen, Append-Only-Log protokolliert und erst danach auf die tatsächlichen Datenstrukturen angewendet wird, die es beschreibt. Der Log-Eintrag erfasst genug Information — die Operation, ihre Parameter und eine Sequenznummer —, um die Änderung entweder vorwärts abzuspielen oder, falls nötig, den Zustand zu rekonstruieren, der ihr voranging, und er gilt erst als vollständig, sobald er auf dauerhaften Speicher geschrieben wurde. Diese Reihenfolge ist die wesentliche Eigenschaft: Da das Log geschrieben und dauerhaft gemacht wird, bevor die entsprechende In-Place-Modifikation bestätigt wird, hinterlässt ein mitten im Schreiben auftretender Absturz eine Spur, die ein Wiederherstellungsprozess nutzen kann, um die unterbrochene Operation abzuschließen oder rückgängig zu machen, statt den Datenspeicher in einem mehrdeutigen, teilweise angewendeten Zustand zu belassen. In Legacy-Systemen zählt dies, weil viele ältere Datenverarbeitungskomponenten Datensätze direkt aktualisieren und annehmen, dass Schreibvorgänge entweder vollständig gelingen oder das System nie mitten in einer Operation abstürzt — eine Annahme, die während ungeplanter Ausfälle, Stromausfällen oder abrupter Prozessterminierung regelmäßig versagt und eine der häufigeren Quellen stiller Datenbeschädigung in alternden Systemen ist. Write-Ahead Logging erstreckt sich auch natürlich auf Legacy-Datenmigrationsanstrengungen: Da es eine dauerhafte, geordnete Aufzeichnung jeder Änderung produziert, erlaubt es einem Migrationsprozess, genau am Punkt der Unterbrechung fortzufahren, statt einen vollständigen Neuvergleich von Quell- und Zieldatensätzen zu erfordern. Die meisten ausgereiften Datenbanken und Messaging-Systeme implementieren diese Technik bereits intern, sodass ihre Anwendung in der Legacy-Modernisierungsarbeit oft eine Frage der Aktivierung und korrekten Konfiguration bestehender WAL-Mechanismen ist, statt neue Logging-Infrastruktur von Grund auf zu bauen.

## How to Apply ◆

> Legacy-Systeme modifizieren Daten oft direkt ohne jeglichen Wiederherstellungsmechanismus, was bedeutet, dass ein Absturz während eines Schreibvorgangs Daten in einem beschädigten, teilweise aktualisierten Zustand zurücklassen kann. Write-Ahead Logging stellt sicher, dass alle Änderungen in einem dauerhaften Log protokolliert werden, bevor sie angewendet werden, was zuverlässige Wiederherstellung nach Fehlschlägen ermöglicht.

- Identifizieren Sie kritische Datenmodifikationspfade im Legacy-System, bei denen ein Fehlschlag während des Schreibprozesses Daten in einem inkonsistenten oder nicht wiederherstellbaren Zustand zurücklassen würde. Dies sind die höchstpriorisierten Kandidaten für Write-Ahead Logging.
- Implementieren Sie ein Append-Only-Log, das jede beabsichtigte Änderung protokolliert, bevor sie auf den primären Datenspeicher angewendet wird. Jeder Log-Eintrag sollte eine eindeutige Sequenznummer, die Operationsdetails, einen Zeitstempel und genug Information enthalten, um die Operation sowohl abzuspielen als auch, falls nötig, rückgängig zu machen.
- Stellen Sie sicher, dass das Log auf dauerhaften Speicher geschrieben wird (auf Festplatte geleert), bevor die Operation gegenüber dem Aufrufer bestätigt wird. Ohne Haltbarkeitsgarantien kann das Log seinen Wiederherstellungszweck nicht erfüllen.
- Implementieren Sie eine Wiederherstellungsprozedur, die nicht committete Log-Einträge nach einem Absturz abspielt. Beim Start liest das System das Log, identifiziert Operationen, die protokolliert, aber nicht als angewendet bestätigt wurden, und spielt sie ab, um den Datenspeicher in einen konsistenten Zustand zu bringen.
- Nutzen Sie Checkpointing, um periodisch einen Punkt im Log zu markieren, an dem alle vorherigen Operationen erfolgreich angewendet wurden. Dies begrenzt die Anzahl der Log-Einträge, die während der Wiederherstellung abgespielt werden müssen, und erlaubt es, alte Log-Segmente zu archivieren oder zu löschen.
- Wenden Sie Write-Ahead Logging auf Datenmigrationsoperationen in Legacy-Systemen an, wo ein Fehlschlag mitten in der Migration Daten aufgeteilt zwischen alten und neuen Speichern in einem inkonsistenten Zustand zurücklassen kann. Das Log bietet einen Mechanismus, um die Migration vom letzten erfolgreichen Checkpoint fortzusetzen.
- Erwägen Sie die Nutzung bestehender WAL-Implementierungen (Datenbank-Transaktionslogs, Apache Kafka, Event-Sourcing-Frameworks) statt eine maßgeschneiderte Lösung zu bauen, da die korrekte Implementierung absturzsicheren Loggings sorgfältige Behandlung von Grenzfällen erfordert.

## Tradeoffs ⇄

> Write-Ahead Logging bietet Absturzwiederherstellung und Datenkonsistenzgarantien, indem sichergestellt wird, dass keine Änderung verloren geht, selbst während unerwarteter Fehlschläge, fügt aber Schreibverstärkung und Speicher-Overhead hinzu.

**Vorteile:**

- Verhindert Datenbeschädigung durch Teilschreibvorgänge, indem sichergestellt wird, dass entweder die vollständige Operation aus dem Log wiederhergestellt oder auf den vorherigen konsistenten Zustand zurückgerollt werden kann.
- Ermöglicht Point-in-Time-Wiederherstellung durch Abspielen des Logs bis zu jeder gewünschten Position, was während Datenmigration und Systemmodernisierung von unschätzbarem Wert ist.
- Bietet eine vollständige Audit-Spur aller Datenmodifikationen, was Debugging, Compliance und Ursachenanalyse unterstützt.
- Unterstützt Replikation und Synchronisation durch Streamen von Log-Einträgen zu sekundären Systemen, was schrittweise Migration von Legacy- zu modernen Datenspeichern ermöglicht.

**Kosten und Risiken:**

- Schreibverstärkung: Jede Datenmodifikation resultiert in mindestens zwei Schreibvorgängen (einer zum Log, einer zum Datenspeicher), was die Performance auf I/O-beschränkten Legacy-Systemen beeinträchtigen kann.
- Log-Speicher wächst kontinuierlich und erfordert Verwaltung — Archivierung, Kompression und Bereinigungsrichtlinien, um Speicherplatzerschöpfung zu verhindern.
- Die korrekte Implementierung absturzsicheren Loggings ist komplex und subtil; Fehler in der Logging- oder Wiederherstellungslogik können genau die Datenbeschädigung verursachen, die sie verhindern sollen.
- Die Wiederherstellungs-Abspielzeit nach einem Absturz hängt von der Menge des nicht komprimierten Logs ab, was den Systemneustart verzögern kann, wenn Checkpointing selten ist.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Write-Ahead Logging Datenverlust und -beschädigung in Legacy-Systemen verhindert.

Ein Legacy-Bestandsverwaltungssystem aktualisiert Lagerbestände, indem es Zeilen in einer Datenbanktabelle direkt modifiziert. Während eines Serverabsturzes, verursacht durch einen Stromausfall, werden mehrere UPDATE-Anweisungen teilweise ausgeführt — die Datenbank committet manche Änderungen, verliert aber andere, die im Buffer Pool waren, aber noch nicht auf Festplatte geleert wurden. Das Ergebnis sind Bestandszahlen, die intern inkonsistent sind: Manche Artikel zeigen negative Lagerbestände, und der Gesamtbestandswert stimmt nicht mehr mit der Summe der einzelnen Artikel überein. Das Team implementiert Write-Ahead Logging, indem es alle Bestandsänderungen durch ein dauerhaftes Log leitet (unter Nutzung von PostgreSQLs eingebautem WAL mit aktivierten synchronen Commits) und verifiziert, dass alle Änderungen vollständig protokolliert sind, bevor Erfolg an die Anwendung zurückgegeben wird. Nach dem nächsten ungeplanten Serverneustart stellt sich die Datenbank automatisch wieder her, indem sie das WAL abspielt, und Bestandszahlen sind perfekt konsistent. Das Team implementiert auch periodisches Checkpointing in 15-Minuten-Intervallen, was die Wiederherstellungszeit auf unter 2 Minuten begrenzt.

Ein Legacy-Data-Warehouse durchläuft eine größere Migration von einer On-Premises-Oracle-Datenbank zu einer cloudbasierten PostgreSQL-Instanz. Die Migration muss inkrementell geschehen, während beide Systeme betriebsbereit bleiben. Das Team implementiert ein Change-Data-Capture-Log, das jeden Schreibvorgang zur Oracle-Datenbank als Append-Only-Eintrag protokolliert. Ein Migrations-Worker liest kontinuierlich aus diesem Log und wendet Änderungen auf die PostgreSQL-Instanz an. Als eine Netzwerkstörung den Migrations-Worker für 45 Minuten unterbricht, setzt er von seiner zuletzt protokollierten Log-Position fort und wendet alle verpassten Änderungen ohne Datenverlust an. Ohne das Write-Ahead-Log hätte das Team einen vollständigen Vergleich beider Datenbanken durchführen müssen, um die verpassten Änderungen zu identifizieren und abzugleichen — ein Prozess, der zuvor 8 Stunden dauerte und sein eigenes Fehlerrisiko einführte.
