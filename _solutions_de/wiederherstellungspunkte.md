---
title: Wiederherstellungspunkte
description: Regelmäßige Sicherung des Systemzustands.
category:
- Operations
problems:
- missing-rollback-strategy
- deployment-risk
- system-outages
- configuration-drift
- data-migration-integrity-issues
- fear-of-change
layout: solution
lang: de
en_slug: restore-points
related_solutions:
- slug: regular-backups
  similarity: 0.85
- slug: rollback-mechanisms
  similarity: 0.85
- slug: disaster-recovery
  similarity: 0.8
- slug: chaos-engineering
  similarity: 0.8
- slug: incident-management
  similarity: 0.8
- slug: backup-and-recovery
  similarity: 0.8
---

## Description

Ein Wiederherstellungspunkt ist ein erfasster Schnappschuss des Systemzustands — eine Datenbank zu einem spezifischen Zeitpunkt, ein virtuelles Maschinenabbild oder eine Konfigurationsbasislinie —, aufgenommen unmittelbar vor einer riskanten Operation wie einer Bereitstellung, Migration oder Konfigurationsänderung, sodass das System in einen bekannt guten Zustand zurückversetzt werden kann, falls diese Operation schiefgeht. Anders als routinemäßige Backups, die nach einem festen Zeitplan erstellt werden, werden Wiederherstellungspunkte bei Bedarf rund um spezifische Änderungsereignisse erstellt und mit Metadaten getaggt, die genau beschreiben, welche Änderung sie ausgelöst hat, was es unkompliziert macht, den korrekten während eines Vorfalls zu identifizieren und zu nutzen. Dies ist besonders wichtig in Legacy-Systemen, die Modernisierung durchlaufen, wo Schemamigrationen, Datentransformationen und Infrastrukturänderungen von Natur aus höheres Risiko tragen als in einem System, das sonst unangetastet bleibt, gerade weil die geänderten Legacy-Codepfade die am wenigsten getesteten und am wenigsten verstandenen Teile des Systems sind. Ohne einen Wiederherstellungspunkt kann eine gescheiterte Migration, die referentielle Integrität mittendrin korrumpiert, zu einem mehrtägigen manuellen Datenreparaturaufwand werden; mit einem wird derselbe Fehler zu einem begrenzten, minutenlangen Rollback, gefolgt von einem zweiten, korrigierten Versuch. Wiederherstellungspunkte fungieren somit als Sicherheitsnetz, spezifisch auf Änderungsereignisse begrenzt, senken das wahrgenommene und tatsächliche Risiko jedes einzelnen Modernisierungsschritts und machen Teams williger, Änderungen zu versuchen, die sie sonst aus Angst vor einem nicht wiederherstellbaren Fehler unbegrenzt aufschieben würden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Erstellen Sie Systemzustands-Schnappschüsse vor jeder bedeutenden Änderung (Bereitstellung, Migration, Konfigurationsaktualisierung)
- Nutzen Sie Point-in-Time-Recovery-Fähigkeiten der Datenbank, um Wiederherstellung zu jedem Moment innerhalb eines Aufbewahrungsfensters zu ermöglichen
- Erfassen Sie Snapshots virtueller Maschinen oder Container als leichtgewichtige Wiederherstellungspunkte für Rollback auf Infrastrukturebene
- Speichern Sie Wiederherstellungspunkte mit Metadaten, die beschreiben, welche Änderung ihre Erstellung ausgelöst hat
- Automatisieren Sie die Erstellung von Wiederherstellungspunkten als Teil von Bereitstellungspipelines, sodass sie nie übersprungen wird
- Testen Sie Wiederherstellung von Wiederherstellungspunkten periodisch, um zu verifizieren, dass sie ein funktionierendes System produzieren
- Definieren Sie Aufbewahrungsrichtlinien, die Speicherkosten mit dem Bedarf an historischen Wiederherstellungsoptionen ausbalancieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht schnelles Rollback, wenn Änderungen unerwartete Probleme in Legacy-Systemen verursachen
- Reduziert das Risiko von Bereitstellungen und Migrationen durch Bereitstellung eines bekannt guten Fallback-Zustands
- Baut Vertrauen für Änderungen an Legacy-Systemen auf
- Bietet einen klaren Wiederherstellungspfad, der Vorfallstress reduziert

**Kosten und Risiken:**
- Wiederherstellungspunkte verbrauchen Speicher, der mit Systemgröße und Änderungshäufigkeit wächst
- Die Wiederherstellung zu einem früheren Punkt kann legitime, nach dem Snapshot erstellte Daten oder Transaktionen verlieren
- Point-in-Time-Recovery erfasst möglicherweise nicht den gesamten Systemzustand (externe Integrationen, Message Queues)
- Teams könnten Wiederherstellungspunkte als Krücke nutzen, statt in richtiges Testen zu investieren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Studierendeninformationssystem einer Universität erforderte eine komplexe Datenbankschemamigration, um neue Einschreibungsfeatures zu unterstützen. Das Team erstellte einen vollständigen Datenbank-Wiederherstellungspunkt und VM-Snapshot vor Beginn der Migration. Als das Migrationsskript mitten in der Migration auf eine unvorhergesehene Constraint-Verletzung stieß, die referentielle Integrität in mehreren Tabellen korrumpierte, stellte das Team den Zustand vor der Migration innerhalb von 20 Minuten wieder her, statt Stunden mit dem Versuch manueller Datenreparatur zu verbringen. Sie behoben das Migrationsskript, testeten es gegen eine Kopie der wiederhergestellten Datenbank und führten es beim zweiten Versuch erfolgreich aus.
