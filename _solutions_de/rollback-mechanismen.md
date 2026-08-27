---
title: Rollback-Mechanismen
description: Fähigkeit, Änderungen rückgängig zu machen und zu einem
  vorherigen stabilen Zustand zurückzukehren.
category:
- Operations
- Process
problems:
- missing-rollback-strategy
- deployment-risk
- frequent-hotfixes-and-rollbacks
- large-risky-releases
- release-instability
- fear-of-change
- complex-deployment-process
- fear-of-failure
- past-negative-experiences
layout: solution
lang: de
en_slug: rollback-mechanisms
related_solutions:
- slug: restore-points
  similarity: 0.85
- slug: canary-releases
  similarity: 0.8
- slug: chaos-engineering
  similarity: 0.8
- slug: regular-backups
  similarity: 0.8
- slug: rolling-updates
  similarity: 0.75
- slug: failover-mechanisms
  similarity: 0.75
---

## Description

Rollback-Mechanismen sind die Bereitstellungs- und Datenebenen-Fähigkeiten, die einem Team erlauben, eine Änderung — eine neue Version, eine Datenbankmigration, eine Konfigurationsaktualisierung — schnell und vorhersagbar zum vorherigen bekannt guten Zustand zurückzuversetzen, statt zu versuchen, ein Problem unter Vorfalldruck vorwärts zu beheben. Diese Fähigkeit aufzubauen bedeutet typischerweise, die Bereitstellungsartefakte der vorherigen Version verfügbar zu halten, jedes Datenbankmigrationsskript mit einem entsprechenden Rollback-Skript zu paaren und Bereitstellungsstrategien wie Blue-Green- oder Canary-Releases zu übernehmen, die das Zurückschalten des Traffics zur vorherigen Version nahezu sofort machen. Legacy-Systemen fehlt diese Fähigkeit häufig vollständig, weil ihre Bereitstellungsprozesse zu einer Zeit etabliert wurden, als Veröffentlichungen selten, manuell und als Einwegoperationen behandelt wurden, was genau der Grund ist, warum jede Bereitstellung gegen ein solches System dazu neigt, als hochriskantes Ereignis behandelt zu werden, das umfangreiche manuelle Verifikation im Voraus erfordert. Die Einführung verlässlicher Rollback-Mechanismen greift diese Dynamik direkt an: Sobald ein Team vertraut, dass jede Bereitstellung innerhalb von Minuten rückgängig gemacht werden kann, sinkt das wahrgenommene Risiko jeder einzelnen Veröffentlichung, was wiederum kleinere, häufigere und daher einzeln sicherere Änderungen ermöglicht — das Gegenteil des großen, seltenen, hochriskanten Veröffentlichungsmusters, in das Legacy-Systeme standardmäßig zu verfallen neigen. Der Mechanismus ist jedoch nicht grenzenlos, da bestimmte Änderungsklassen, wie Datenformatmigrationen oder externe API-Vertragsänderungen, von Natur aus schwer oder unmöglich sauber rückgängig zu machen sind, was bedeutet, dass Rollback-Fähigkeit Änderung für Änderung bewertet werden muss, statt universell als vorhanden angenommen zu werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Gestalten Sie jede Bereitstellung so, dass sie reversibel ist, indem Sie die Artefakte und Konfiguration der vorherigen Version aufbewahren
- Implementieren Sie Datenbankmigrations-Rollback-Skripte neben Vorwärtsmigrationen
- Nutzen Sie Blue-Green- oder Canary-Bereitstellungsstrategien, die sofortiges Umschalten des Traffics zur vorherigen Version ermöglichen
- Automatisieren Sie Rollback-Verfahren, sodass sie unter Vorfalldruck schnell ausgeführt werden können
- Definieren Sie Rollback-Entscheidungskriterien (Fehlerratenschwellen, Latenzanstiege) und ermächtigen Sie Teams, ohne Managementgenehmigung zu handeln
- Testen Sie Rollback-Verfahren als Teil der Bereitstellungspipeline, nicht nur die Vorwärtsbereitstellung
- Halten Sie Rollback-Artefakte für einen definierten Aufbewahrungszeitraum nach jeder Bereitstellung verfügbar

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert das Risiko und die Auswirkung fehlgeschlagener Bereitstellungen dramatisch
- Ermöglicht schnelleren Bereitstellungsrhythmus durch Bereitstellung eines Sicherheitsnetzes
- Reduziert die Vorfalldauer durch Bereitstellung eines schnellen Pfads zu einem bekannt guten Zustand
- Baut Teamvertrauen auf, Änderungen an Legacy-Systemen häufiger bereitzustellen

**Kosten und Risiken:**
- Datenbank-Rollback-Skripte müssen sorgfältig entworfen werden, um Datenverlust zu vermeiden
- Manche Änderungen (Datenformatmigrationen, API-Vertragsänderungen) sind schwer rückgängig zu machen
- Die Aufrechterhaltung von Rollback-Fähigkeit fügt jeder Bereitstellung Aufwand hinzu
- Häufige Abhängigkeit von Rollback kann tiefere Qualitätsprobleme signalisieren, die adressiert werden müssen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Finanzdienstleistungsunternehmen stellte Updates für seine Legacy-Handelsplattform einmal pro Quartal bereit, weil jede Bereitstellung riskant war und keine Rollback-Fähigkeit hatte. Nach der Implementierung automatisierter Rollback-Mechanismen, einschließlich Datenbankmigrations-Umkehr, Artefakt-Versionierung und Load-Balancer-Traffic-Umschaltung, konnte das Team jede Bereitstellung innerhalb von fünf Minuten rückgängig machen. Dieses Sicherheitsnetz erlaubte dem Team, die Bereitstellungshäufigkeit auf wöchentlich zu erhöhen, wobei es im ersten Quartal drei problematische Veröffentlichungen erfasste und zurücksetzte, während die durchschnittliche Größe und das Risiko jeder Bereitstellung reduziert wurden.
