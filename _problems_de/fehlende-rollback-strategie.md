---
title: Fehlende Rollback-Strategie
description: Es gibt keine getestete Methode, ein Deployment rückgängig zu machen,
  falls etwas schiefgeht, was das Risiko erhöht.
category:
- Code
- Management
- Process
related_problems:
- slug: deployment-risk
  similarity: 0.8
- slug: immature-delivery-strategy
  similarity: 0.65
- slug: frequent-hotfixes-and-rollbacks
  similarity: 0.65
- slug: complex-deployment-process
  similarity: 0.65
- slug: history-of-failed-changes
  similarity: 0.65
- slug: manual-deployment-processes
  similarity: 0.65
solutions:
- blue-green-canary-deployments
- ci-cd-pipeline
- feature-flags
- backup-and-recovery
- disaster-recovery
- regular-backups
- restore-points
- rollback-mechanisms
- stress-testing
- write-ahead-logging
- emergency-drills
- incident-response-measures
- risk-quantification
layout: problem
lang: de
en_slug: missing-rollback-strategy
---

## Description

Deployment-Risiko tritt auf, wenn Teams Systeme deployen, ohne eine zuverlässige, getestete Methode zu haben, um schnell zu einem vorherigen funktionierenden Zustand zurückzukehren, wenn Probleme auftreten. Dies schafft erhebliches Risiko während Deployments, da alle nach dem Deployment entdeckten Probleme nur durch Vorwärtskorrektur behoben werden können, was erhebliche Zeit in Anspruch nehmen und zu verlängerten Ausfällen führen kann. Das Fehlen von Rollback-Fähigkeiten führt oft zu Deployment-Angst, längeren Vorfallslösungszeiten und größerer Auswirkung, wenn Deployments schiefgehen.

## Indicators ⟡

- Deployment-Verfahren, die nur Vorwärts-Deployment-Schritte dokumentieren
- Datenbankmigrationsskripte ohne entsprechende Rollback-Skripte
- Infrastrukturänderungen, die schwierig oder unmöglich rückgängig zu machen sind
- Deployment-Angst und Zurückhaltung, während der Geschäftszeiten zu deployen
- Notfallreaktionspläne, die Vorwärtskorrektur als einzige Option voraussetzen
- Keine Tests von Rollback-Verfahren während der Deployment-Planung
- Konfigurationsänderungen, die vorherige Einstellungen ohne Backup überschreiben

## Symptoms ▲

- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Wenn Deployments schiefgehen und nicht schnell rückgängig gemacht werden können, dauern Vorfälle länger und haben größere Auswirkung.
- [Große, riskante Releases](grosse-riskante-releases.md)
<br/>  Deployment-Angst durch fehlendes Rollback führt dazu, dass Teams Änderungen in weniger, größere Releases bündeln, die noch riskanter sind.
- [Release-Angst](release-angst.md)
<br/>  Ohne getestete Rollback-Strategie wird jedes Deployment zu einem Hochrisiko-Ereignis, da Probleme nicht leicht rückgängig gemacht werden können.

## Causes ▼

- [Manuelle Deployment-Prozesse](manuelle-deployment-prozesse.md)
<br/>  Manuelle Deployment-Workflows erschweren die Implementierung und das Testen zuverlässiger Rollback-Verfahren.
- [Komplexer Deployment-Prozess](komplexer-deployment-prozess.md)
<br/>  Übermäßig komplexe Deployment-Prozesse machen es unpraktikabel, Rollback-Schritte für jede Komponente zu definieren und zu testen.
- [Unausgereifte Auslieferungsstrategie](unausgereifte-auslieferungsstrategie.md)
<br/>  Organisationen mit unausgereiften Auslieferungspraktiken fehlt oft die Disziplin, Rollback-Strategien als Teil des Deployments zu planen und zu testen.

## Detection Methods ○

- Überprüfung der Deployment-Dokumentation auf Abdeckung von Rollback-Verfahren
- Audit von Datenbankmigrationsskripten auf Vorhandensein von Rollback-/Down-Migrationen
- Testen von Rollback-Verfahren in Staging-Umgebungen als Teil der Deployment-Planung
- Bewertung von Infrastruktur-Provisionierungswerkzeugen auf Zustandsmanagement und Rollback-Fähigkeiten
- Befragung von Deployment-Teams zum Vertrauen in Rollback-Optionen
- Überprüfung von Vorfallreaktionsverfahren auf Rollback- vs. Vorwärtskorrektur-Entscheidungsbäume
- Untersuchung von Deployment-Werkzeugen auf eingebaute Rollback-Funktionalität
- Analyse historischer Vorfalldaten für Fälle, in denen Rollback die Auswirkung reduziert hätte

## Examples

Eine E-Commerce-Plattform deployt ein neues Zahlungsverarbeitungsfeature während eines routinemäßigen Freitagabend-Releases. Das Deployment umfasst Datenbankschemaänderungen, die neue Spalten hinzufügen und bestehende Constraints modifizieren. Zwei Stunden nach dem Deployment beginnen Kundenberichte über fehlgeschlagene Zahlungsverarbeitung einzugehen, die Bestellungen blockiert. Das Team entdeckt einen kritischen Fehler in der neuen Zahlungslogik, der alle Transaktionen betrifft. Sie erkennen jedoch, dass sie kein Rollback durchführen können, weil die Datenbankmigrationen irreversibel sind – sie fügten erforderliche Spalten hinzu, die ohne Datenverlust nicht sicher entfernt werden können. Das Team ist gezwungen, das gesamte Wochenende mit der Fehlerbehebung des Zahlungsproblems in Produktion zu verbringen, während die E-Commerce-Website Umsatz durch fehlgeschlagene Transaktionen verliert. Eine ordentliche Rollback-Strategie mit reversiblen Datenbankänderungen hätte den Service innerhalb von Minuten statt Tagen wiederherstellen können.
