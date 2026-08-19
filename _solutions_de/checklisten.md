---
title: Checklisten
description: Systematische Abarbeitung von Schritten und Anforderungen.
category:
- Process
problems:
- inconsistent-quality
- quality-blind-spots
- poor-documentation
- inadequate-code-reviews
- complex-deployment-process
- rushed-approvals
- implementation-starts-without-design
- inadequate-initial-reviews
- inconsistent-execution
- inconsistent-onboarding-experience
- review-process-breakdown
- reviewer-anxiety
- reviewer-inexperience
- unproductive-meetings
- code-review-inefficiency
- conflicting-reviewer-opinions
- insufficient-code-review
- superficial-code-reviews
- review-process-avoidance
layout: solution
lang: de
en_slug: checklists
related_solutions:
- slug: runbooks
  similarity: 0.8
- slug: portability-checklists
  similarity: 0.75
- slug: incident-management
  similarity: 0.75
- slug: ci-cd-pipeline
  similarity: 0.75
- slug: restore-points
  similarity: 0.7
- slug: blameless-postmortems
  similarity: 0.7
---

## Description

Eine Checkliste ist eine kurze, explizite, geordnete Liste der Schritte oder Anforderungen, die für einen gegebenen repetitiven und fehleranfälligen Prozess abgeschlossen werden müssen, genutzt als externes Gedächtnishilfsmittel, das nicht davon abhängt, dass sich eine Einzelperson unter Druck an jeden Schritt korrekt erinnert. Sie funktioniert, indem sie stillschweigende Erwartungen darüber, „wie das getan werden soll", in ein sichtbares Artefakt verwandelt, dem konsistent gefolgt werden kann, unabhängig davon, wer die Aufgabe durchführt oder wie erfahren sie ist. In Legacy-Systemen, wo sich Deployment-Schritte, Review-Kriterien und Vorfallprozeduren oft als ungeschriebene Konventionen angehäuft haben, die nur wenigen langjährigen Teammitgliedern bekannt sind, sind Checklisten ein kostengünstiger Weg, dieses implizite Wissen zu externalisieren, bevor es durch Fluktuation verloren geht. Sie sind besonders effektiv gegen Auslassungsfehler — die Fehlerklasse, bei der jemand einfach einen notwendigen Schritt vergisst, statt einen Schritt falsch durchzuführen —, was genau der Fehlermodus ist, der in komplexen, selten geänderten Legacy-Prozessen dominiert. Weil Checklisten keine Tooling-Investition erfordern, um eingeführt zu werden, sind sie oft der erste, sofort umsetzbare Schritt zur Stabilisierung eines chaotischen Prozesses und können später als die Spezifikation dienen, aus der automatisierte Prüfungen Element für Element gebaut werden. Ihr Wert hängt jedoch vollständig von aktiver Pflege ab: eine Checkliste, die nicht aktualisiert wird, während sich der zugrunde liegende Prozess ändert, verwandelt sich still in eine falsche Vertrauensquelle statt eines Schutzmechanismus.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie repetitive, fehleranfällige Prozesse im Entwicklungslebenszyklus (Deployments, Code-Reviews, Vorfallreaktion), die von Checklisten profitieren würden
- Erstellen Sie prägnante Checklisten mit klaren, umsetzbaren Punkten statt vager Empfehlungen
- Integrieren Sie Checklisten in bestehende Workflows wie Pull-Request-Vorlagen, Deployment-Skripte oder Vorfall-Runbooks
- Überprüfen und aktualisieren Sie Checklisten regelmäßig basierend auf neuen Befunden, Postmortems und sich ändernden Anforderungen
- Halten Sie Checklisten kurz genug, um praktisch zu sein (maximal 10-15 Punkte), während kritische Schritte abgedeckt werden
- Unterscheiden Sie zwischen verpflichtenden Punkten, die abgeschlossen werden müssen, und optionalen Punkten, die situativ sind

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verringert Auslassungsfehler, indem erforderliche Schritte explizit gemacht werden
- Stellt Konsistenz über Teammitglieder hinweg sicher, die denselben Prozess durchführen
- Erfasst institutionelles Wissen in einer Form, die Teamfluktuation überlebt
- Kostengünstige Praxis, die sofort ohne Tooling-Änderungen übernommen werden kann

**Kosten und Risiken:**
- Checklisten können veralten und an Relevanz verlieren, wenn sie nicht aktiv gepflegt werden
- Mechanische Checkbox-Einhaltung ohne echtes Engagement bietet falsches Vertrauen
- Übermäßig detaillierte Checklisten verlangsamen Prozesse und ermutigen zu Abkürzungen
- Ersetzt keine Expertise und Urteilsvermögen für komplexe Entscheidungen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Systemteam erlebte wiederkehrende Deployment-Fehler, weil verschiedene Teammitglieder Deployments unterschiedlich durchführten, jeder unterschiedliche Schritte vergessend. Das Team erstellte eine Deployment-Checkliste, die Vor-Deployment-Validierung, Backup-Verifikation, Migrationsausführung, Smoke-Testing und Rollback-Kriterien abdeckte. Die Checkliste wurde in ihr Deployment-Skript als eine Reihe von Bestätigungsaufforderungen eingebettet. Deployment-Fehler sanken von durchschnittlich zwei pro Monat auf einen pro Quartal. Die Checkliste wurde außerdem zum Ausgangspunkt für die Automatisierung von Deployment-Schritten, wobei jeder Punkt schließlich durch eine automatisierte Prüfung ersetzt wurde.
