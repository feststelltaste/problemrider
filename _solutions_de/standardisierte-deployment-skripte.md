---
title: Standardisierte Deployment-Skripte
description: Erstellung einheitlicher Skripte für Deployment und
  Konfiguration über verschiedene Plattformen hinweg.
category:
- Operations
- Process
problems:
- complex-deployment-process
- manual-deployment-processes
- deployment-environment-inconsistencies
- deployment-risk
- configuration-drift
- immature-delivery-strategy
- frequent-hotfixes-and-rollbacks
layout: solution
lang: de
en_slug: standardized-deployment-scripts
related_solutions:
- slug: platform-independent-scripting-languages
  similarity: 0.8
- slug: platform-independent-configuration-management
  similarity: 0.75
- slug: platform-independent-configuration-files
  similarity: 0.75
- slug: continuous-integration-and-delivery
  similarity: 0.75
- slug: automated-migration-tools
  similarity: 0.75
- slug: cross-platform-build-scripts
  similarity: 0.75
---

## Description

Standardisierte Deployment-Skripte ersetzen Ad-hoc-, umgebungsspezifische Deployment-Prozeduren — manuelle SSH-Befehle, umgebungsspezifische Shell-Skripte, wiki-dokumentierte Schritte — durch ein einziges, parametrisiertes Automatisierungsskript oder Playbook, gebaut mit Werkzeugen wie Ansible oder Terraform, das identisch über Entwicklung, Staging und Produktion läuft. Legacy-Systeme sammeln häufig Deployment-Prozesse an, die subtil zwischen Umgebungen abweichen, weil jede ad hoc von wem auch immer, der gerade Bereitschaftsdienst hatte, gepatcht wurde, und die resultierende Inkonsistenz ist eine wiederkehrende Quelle umgebungsspezifischer Vorfälle, die schwer zu reproduzieren und noch schwerer zu verhindern sind. Durch das einmalige Erfassen der Deployment-Logik, die Parametrisierung nur dessen, was sich genuin zwischen Umgebungen unterscheidet, und die Speicherung des Ergebnisses in der Versionskontrolle neben dem Anwendungscode verwandelt diese Praxis, was stilles Wissen war, verstreut über Skripte und Erinnerung, in ein explizites, überprüfbares und wiederholbares Artefakt. Dies zählt für Legacy-Modernisierung speziell, weil inkonsistentes Deployment oft das ist, was jede Änderung an einem Legacy-System überhaupt erst riskant erscheinen lässt — wenn Deployment unvorhersehbar ist, erbt jede andere Verbesserung diese Unvorhersehbarkeit. Die Vorabkosten sind der Aufwand, die Unterschiede über Umgebungen hinweg zu einem kohärenten Skript zu versöhnen, und die resultierende Automatisierung braucht weiterhin Betriebspersonal, das das Tooling gut genug versteht, um einen Fehlschlag zu diagnostizieren, wenn das Skript selbst kaputtgeht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Dokumentieren Sie den aktuellen Deployment-Prozess für jede Zielumgebung und erfassen Sie manuelle Schritte, Skripte und stilles Wissen
- Identifizieren Sie Gemeinsamkeiten und Unterschiede über Deployment-Ziele hinweg, um eine einheitliche Skriptstruktur zu gestalten
- Erstellen Sie Deployment-Skripte mit plattformübergreifenden Werkzeugen wie Ansible, Terraform oder Python-basierter Automatisierung
- Parametrisieren Sie umgebungsspezifische Werte, sodass dasselbe Skript über Entwicklung, Staging und Produktion funktioniert
- Beziehen Sie Vor-Deployment-Validierungsprüfungen (Service-Gesundheit, Konfigurationskorrektheit, Speicherplatz) in die Skripte ein
- Fügen Sie jedem Deployment-Skript Rollback-Fähigkeiten hinzu, sodass fehlgeschlagene Deployments schnell rückgängig gemacht werden können
- Speichern Sie Deployment-Skripte in der Versionskontrolle neben dem Anwendungscode und unterziehen Sie sie Code-Review

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Stellt sicher, dass Deployments über alle Umgebungen hinweg konsistent und wiederholbar sind
- Reduziert menschlichen Fehler durch Beseitigung manueller Deployment-Schritte
- Macht Deployment-Wissen explizit und versionskontrolliert statt still
- Ermöglicht schnellere Notfallwiederherstellung durch automatisierte Neubereitstellung

**Kosten und Risiken:**
- Der anfängliche Aufwand zur Standardisierung von Skripten über heterogene Umgebungen hinweg kann erheblich sein
- Übermäßig starre Skripte könnten Grenzfälle nicht handhaben, die manuelle Prozesse informell berücksichtigten
- Skriptfehlschläge in Produktion erfordern, dass Betriebspersonal das Automatisierungstooling versteht
- Die Pflege von Skripten erfordert laufenden Aufwand, während sich Anwendung und Infrastruktur weiterentwickeln

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Medienunternehmen setzte sein Legacy-CMS in drei verschiedenen Umgebungen mit einer Kombination aus manuellen SSH-Befehlen, maßgeschneiderten Bash-Skripten und einer Wiki-Seite mit Deployment-Anweisungen ein. Jedes Deployment dauerte 45 Minuten, und der Prozess unterschied sich subtil zwischen Umgebungen, was monatliche Vorfälle verursachte. Das Team vereinheitlichte den Prozess in Ansible-Playbooks mit umgebungsspezifischen Variablendateien. Deployments wurden zu einem einzigen Befehl, unabhängig von der Zielumgebung, die Abschlusszeit sank auf acht Minuten, und deploymentbezogene Vorfälle sanken um 85 %. Die Playbooks dienten auch als lebende Dokumentation der Deployment-Architektur.
