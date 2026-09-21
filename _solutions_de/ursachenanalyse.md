---
title: Ursachenanalyse
description: Systematische Analyse der Ursachen von Ausfällen.
category:
- Process
problems:
- constant-firefighting
- high-defect-rate-in-production
- partial-bug-fixes
- regression-bugs
- slow-incident-resolution
- delayed-issue-resolution
- increased-error-rates
- blame-culture
layout: solution
lang: de
en_slug: root-cause-analysis
related_solutions:
- slug: incident-management
  similarity: 0.85
- slug: error-logs
  similarity: 0.8
- slug: error-reporting-and-analysis
  similarity: 0.8
- slug: runbooks
  similarity: 0.75
- slug: chaos-engineering
  similarity: 0.75
- slug: blameless-postmortems
  similarity: 0.75
---

## Description

Ursachenanalyse ist eine strukturierte Untersuchungstechnik — üblicherweise unter Nutzung von Methoden wie den „5 Warums" oder einem Fischgrätendiagramm —, die nach einem Produktionsvorfall durchgeführt wird, um die Kausalkette vom sichtbaren Symptom zurück zum zugrunde liegenden Zustand zu verfolgen, der ihn tatsächlich produzierte, statt bei der am einfachsten zu behebenden unmittelbaren Ursache anzuhalten. Ihre Ausgabe ist eine klare Trennung zwischen dem Symptom, das den Vorfall auslöste, den beitragenden Faktoren, die ihn möglich machten, und der wahren Grundursache, zusammen mit konkreten Folgeaktionen, die spezifischen Eigentümern zugewiesen sind. Diese Unterscheidung ist kritisch in Legacy-Systemen, wo der Weg des geringsten Widerstands während eines Vorfalls fast immer ist, einen schnellen, lokalisierten Patch anzuwenden und weiterzumachen, was dazu neigt, den tatsächlichen zugrunde liegenden Defekt — eine nicht indizierte Tabelle, eine Race Condition, eine in vor einem Jahrzehnt geschriebenem Code eingebackene Annahme — unangetastet zu lassen und frei, oft wiederholt, in leicht unterschiedlichen Erscheinungsformen wieder aufzutauchen. Da Legacy-Systeme Vorfälle ansammeln, deren Grundursachen häufig über scheinbar unzusammenhängende Symptome geteilt werden, offenbart eine disziplinierte Ursachenanalyse-Praxis über die Zeit systemische Muster, die gelegentlich zeigen, dass eine einzelne strukturelle Behebung eine ganze Kategorie wiederkehrender Feuerwehreinsätze beseitigt, die unverhältnismäßigen Engineering-Aufwand verbraucht hatten. Gut durchgeführt, mit funktionsübergreifender Beteiligung und ohne Schuldzuweisung, verwandelt sie die betrieblichen Kosten von Produktionsvorfällen in einen sich anhäufenden Bestand organisatorischen Wissens darüber, wie das Legacy-System genau zu scheitern neigt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Führen Sie nach jedem bedeutenden Produktionsvorfall eine strukturierte Ursachenanalyse mittels Techniken wie den „5 Warums" oder Fischgrätendiagrammen durch
- Unterscheiden Sie zwischen Symptomen, beitragenden Faktoren und wahren Grundursachen, bevor Sie Fixes implementieren
- Beziehen Sie funktionsübergreifende Teilnehmer ein, einschließlich Entwickler, Operations und Geschäfts-Stakeholder
- Dokumentieren Sie Befunde in einer gemeinsamen Wissensdatenbank, zugänglich für alle Teammitglieder
- Verfolgen Sie Grundursachen-Kategorien über die Zeit, um systemische Muster zu identifizieren (z. B. die meisten Vorfälle durch Legacy-Datenbankprobleme verursacht)
- Erstellen Sie umsetzbare Folgepunkte mit Eigentümern und Fristen für jede identifizierte Grundursache
- Überprüfen Sie die Wirksamkeit implementierter Fixes, indem Sie prüfen, ob ähnliche Vorfälle wiederkehren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Durchbricht den Zyklus wiederkehrender Vorfälle, indem Ursachen statt Symptome adressiert werden
- Baut organisatorisches Wissen über Legacy-System-Fehlermodi auf
- Treibt systemische Verbesserungen an, die die gesamte Vorfallhäufigkeit reduzieren
- Schafft eine Feedback-Schleife zwischen Produktionsbetrieb und Entwicklungspraktiken

**Kosten und Risiken:**
- Analyse nimmt Zeit von Feature-Entwicklung und unmittelbarer Feuerwehr weg
- Schuldorientierte Analyse entmutigt ehrliche Beteiligung und verbirgt echte Ursachen
- Ursachenanalyse kann zu Analyselähmung führen, wenn sie nicht zeitbegrenzt ist
- Nicht alle Grundursachen sind praktisch in Legacy-Systemen zu beheben, was Risikoakzeptanzentscheidungen erfordert

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Logistikunternehmen erlebte wöchentliche Ausfälle in seinem Legacy-Routingsystem. Jedes Mal wendete das Team einen schnellen Fix an und machte weiter. Nach der Einführung verpflichtender Ursachenanalyse entdeckte das Team, dass alle Vorfälle auf eine einzige Datenbanktabelle zurückgingen, der eine ordentliche Indizierung für die Abfragemuster eines vor zwei Jahren hinzugefügten Features fehlte. Die Abfrage funktionierte bei geringen Datenvolumina gut, verschlechterte sich aber, während die Tabelle wuchs. Das Hinzufügen eines einzigen Index beseitigte eine gesamte Klasse von Vorfällen, die im Vorjahr Hunderte von Engineering-Stunden in Feuerwehreinsätzen verbraucht hatten.
