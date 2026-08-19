---
title: Portabilitäts-Checklisten
description: Erstellung von Checklisten zur Prüfung der Portabilität auf
  verschiedene Systeme und Plattformen.
category:
- Process
problems:
- technology-lock-in
- vendor-lock-in
- deployment-environment-inconsistencies
- quality-blind-spots
- inconsistent-quality
- poor-documentation
layout: solution
lang: de
en_slug: portability-checklists
related_solutions:
- slug: checklists
  similarity: 0.75
- slug: platform-independence
  similarity: 0.75
- slug: documentation-of-compatibility-requirements
  similarity: 0.75
- slug: compatibility-testing
  similarity: 0.75
- slug: abstraction-layers
  similarity: 0.7
- slug: cross-platform-build-tools
  similarity: 0.7
---

## Description

Eine Portabilitäts-Checkliste ist eine strukturierte, gepflegte Liste von Plattformabhängigkeitsrisiken — betriebssystemspezifische Pfadtrenner, Zeilenendenkonventionen, Zeichenkodierungsannahmen, Byte-Reihenfolge, datenbankspezifisches SQL, Dateisystemannahmen —, die Reviewer während Code-Review, Architektur-Review und Technologieauswahl durcharbeiten, um Portabilitätsprobleme zu erfassen, bevor sie Produktion erreichen. Es ist eine Lösung auf Prozessebene statt einer technischen: Sie behebt nichts von sich aus, sondern erzwingt eine systematische, wiederholbare Prüfung, die sonst davon abhängen würde, dass sich ein einzelner Reviewer zufällig an die relevante Falle erinnert. Dies zählt für die Legacy-Modernisierung, weil in altem Code eingebackene Portabilitätsannahmen normalerweise unsichtbar sind, bis das System tatsächlich auf eine neue Plattform umgezogen wird, und zu diesem Zeitpunkt ist die Behebung weit teurer, als die Annahme während des ursprünglichen Code-Reviews zu erfassen. Eine Checkliste fungiert auch als institutionelles Gedächtnis: Sie erfasst die Plattformeigenheiten, an denen sich ein Team in der Vergangenheit verbrannt hat, und gibt dieses Wissen an neue Teammitglieder weiter, die nicht anwesend waren, als der ursprüngliche Portabilitätsvorfall auftrat. Ihre Hauptbeschränkung ist, dass eine Checkliste nur so gut ist wie die Disziplin dahinter — manuelle Checklistenpunkte riskieren, unter Zeitdruck zu einem oberflächlichen Abnicken zu werden, und Portabilitätsprobleme außerhalb der aufgeführten Kategorien werden schlicht übersehen, egal wie gründlich die aufgeführten Punkte geprüft werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie eine Portabilitäts-Checkliste, die Schlüsselbereiche abdeckt: Betriebssystemabhängigkeiten, Datenbankkompatibilität, Dateisystemannahmen, Netzwerkkonfiguration und externe Dienstintegrationen
- Beziehen Sie Prüfungen für plattformspezifische Konstrukte ein wie Pfadtrenner, Zeilenenden, Zeichenkodierungen und Byte-Reihenfolge
- Integrieren Sie die Checkliste in Code-Review-Prozesse, sodass Portabilität vor dem Zusammenführen von Änderungen verifiziert wird
- Überprüfen und aktualisieren Sie die Checkliste, wann immer eine neue Zielplattform hinzugefügt oder ein Portabilitätsproblem entdeckt wird
- Automatisieren Sie Checklistenpunkte, wo möglich, durch Hinzufügen von Linting-Regeln oder statischen Analyseprüfungen
- Verwenden Sie die Checkliste während Architektur-Reviews und Technologieauswahlentscheidungen, um sicherzustellen, dass neue Komponenten Portabilitätsanforderungen erfüllen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet einen systematischen Ansatz zur Identifikation von Portabilitätsrisiken, bevor sie Produktion erreichen
- Dient als institutionelles Wissen, das Personalfluktuation überdauert
- Schafft Konsistenz darin, wie Portabilität über Teams und Projekte hinweg bewertet wird
- Günstige Praxis, die sofort ohne Tooling-Änderungen übernommen werden kann

**Kosten und Risiken:**
- Checklisten können veralten, wenn sie nicht regelmäßig gepflegt und aktualisiert werden
- Checkbox-Compliance kann ohne echte Untersuchung oberflächlich werden
- Manuelle Checklisten skalieren nicht gut für große Codebasen oder häufige Änderungen
- Kann ein falsches Sicherheitsgefühl erzeugen, wenn wichtige Portabilitätsaspekte nicht abgedeckt sind

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Softwareberatungsunternehmen entwickelte Anwendungen für Kunden mit vielfältiger Infrastruktur. Nach wiederholten Portabilitätsproblemen während Bereitstellungen erstellten sie eine Portabilitäts-Checkliste, die 40 Punkte über sechs Kategorien abdeckte. Die Checkliste wurde in ihre Pull-Request-Vorlage integriert, sodass jede Änderung dagegen bewertet wurde. Über ein Jahr sank die Anzahl der Portabilitätsfehler zur Bereitstellungszeit um 70 %. Die Checkliste wurde außerdem zu einem wertvollen Onboarding-Werkzeug, das neuen Entwicklern half, die Arten von Plattformannahmen zu verstehen, die zu vermeiden sind. Als sie später 25 der 40 Prüfungen als CI-Pipeline-Regeln automatisierten, wurden die verbleibenden manuellen Punkte fokussierter und bedeutsamer.
