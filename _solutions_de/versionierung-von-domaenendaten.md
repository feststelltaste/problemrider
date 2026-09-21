---
title: Versionierung von Domänendaten
description: Nachverfolgung und Wiederherstellung von Änderungen an domänenspezifischen
  Daten.
category:
- Database
problems:
- silent-data-corruption
- data-migration-integrity-issues
- insufficient-audit-logging
- schema-evolution-paralysis
- debugging-difficulties
layout: solution
lang: de
en_slug: domain-data-versioning
related_solutions:
- slug: timestamping
  similarity: 0.65
- slug: versioning-scheme
  similarity: 0.65
- slug: evolutionary-database-design
  similarity: 0.6
- slug: data-integrity
  similarity: 0.6
- slug: continuous-data-verification
  similarity: 0.6
- slug: write-ahead-logging
  similarity: 0.6
---

## Description

Versionierung von Domänendaten erfasst die vollständige Historie von Änderungen an kritischen Geschäftsentitäten — wer was wann und oft warum geändert hat — mittels Mechanismen wie temporalen oder Audit-Tabellen, Entitätsebenen-Versionierung oder Event Sourcing, statt jedes Update den vorherigen Zustand der Entität still überschreiben zu lassen. Viele Legacy-Systeme wurden nur mit einem „aktueller Zustand"-Modell gebaut, da Audit-Trails vor Jahrzehnten keine Priorität waren, was bedeutet, dass, sobald ein Wert überschrieben wurde, es keine Möglichkeit gibt wiederherzustellen, was er einst war oder wann oder warum er sich änderte — eine Lücke, die akut schmerzhaft wird, in dem Moment, in dem ein Streitfall, ein Audit oder eine unerklärliche Datenanomalie die Rekonstruktion einer Historie erfordert, die nie erfasst wurde. Versionierung nachträglich hinzuzufügen verwandelt ein undurchsichtiges Einzel-Snapshot-Datenmodell in eines, in dem jeder vergangene Zustand rekonstruiert und mit dem aktuellen verglichen werden kann, was direkt das Debugging stiller Datenkorruption unterstützt und Compliance während regulatorischer oder rechtlicher Überprüfung nachweist. Es ist auch während Datenmigrationen unverhältnismäßig wertvoll, da der Vergleich versionierter Quell- und Zielhistorien eine weit stärkere Korrektheitsprüfung bietet als der Vergleich finaler Snapshots allein. Der Tradeoff ist ein echter Anstieg des Speichervolumens und Schreiboverheads bei jeder Modifikation, zusammen mit zusätzlicher Komplexität in der Datenzugriffsschicht für die Abfrage historischer Zustände, sodass Aufbewahrungsrichtlinien und Abfragemuster bewusst abgegrenzt werden müssen, statt standardmäßig unbegrenzt alles zu versionieren.

## How to Apply ◆

- Implementieren Sie temporale Tabellen oder Audit-Tabellen, die jede Änderung an kritischen Domänenentitäten zusammen mit Zeitstempeln, Nutzern und Änderungsgründen erfassen.
- Fügen Sie Domänenobjekten Versionierung hinzu, sodass der aktuelle Zustand und die vollständige Historie jeder Entität verfügbar sind.
- Nutzen Sie Event Sourcing für kritische Geschäftsentitäten, bei denen die Fähigkeit, den Zustand zu jedem beliebigen Zeitpunkt zu rekonstruieren, wertvoll ist.
- Bauen Sie Werkzeuge zum Vergleich von Entitätsversionen und zur Identifikation, wann und warum sich Daten änderten, was Ursachenanalyse unterstützt.
- Stellen Sie sicher, dass Datenversionierung Migrationen und Massenaktualisierungen abdeckt, nicht nur einzelne Datensatzänderungen.
- Definieren Sie Aufbewahrungsrichtlinien für historische Datenversionen, um das Speicherwachstum zu verwalten.

## Tradeoffs ⇄

**Vorteile:**
- Ermöglicht Auditierung und Compliance, indem eine vollständige Historie von Datenänderungen bereitgestellt wird.
- Unterstützt Debugging, indem die Rekonstruktion des Systemzustands zu jedem vergangenen Zeitpunkt erlaubt wird.
- Bietet ein Sicherheitsnetz für Datenkorrekturen: Falsche Änderungen können identifiziert und rückgängig gemacht werden.
- Erleichtert die Validierung von Datenmigrationen durch den Vergleich von Quell- und Zielversionen.

**Kosten:**
- Das Speichern jeder Version jeder Domänenentität erhöht die Speicheranforderungen erheblich.
- Fügt jeder Datenmodifikationsoperation Schreiboverhead hinzu.
- Das Abfragen historischer Daten fügt der Datenzugriffsschicht Komplexität hinzu.
- Versionierung nachträglich in ein Legacy-System ohne bestehenden Audit-Trail einzubauen erfordert Schemaänderungen und Migration.

## How It Could Be

Ein Legacy-Vertragsmanagementsystem hat keinen Audit-Trail, was es unmöglich macht zu bestimmen, wann oder warum die Bedingungen eines Vertrags modifiziert wurden. Nach einem Streitfall, bei dem ein Kunde behauptet, seine Preisgestaltung sei ohne Autorisierung geändert worden, fügt das Team Versionierung von Domänendaten mittels temporaler Tabellen hinzu. Jede Vertragsmodifikation wird jetzt mit einem Zeitstempel, dem Nutzer, der die Änderung vorgenommen hat, und den vorherigen Werten erfasst. Als sechs Monate später ein ähnlicher Streitfall entsteht, kann das Team die exakte Änderungshistorie zeigen, wer sie autorisiert hat und wann sie geschahen. Das Versionierungssystem erweist sich auch während einer Datenmigration als unschätzbar, wo das Team Versionshistorien nutzt, um zu verifizieren, dass die Migration alle Vertragsbedingungen korrekt bewahrt hat.
