---
title: Erhöhte manuelle Arbeit
description: Entwickler verbringen Zeit mit repetitiven Aufgaben, die automatisiert
  werden sollten, was die für tatsächliche Entwicklungsarbeit verfügbare Zeit verringert.
category:
- Code
- Process
related_problems:
- slug: increased-manual-testing-effort
  similarity: 0.75
- slug: inefficient-processes
  similarity: 0.7
- slug: manual-deployment-processes
  similarity: 0.65
- slug: inefficient-development-environment
  similarity: 0.65
- slug: extended-research-time
  similarity: 0.65
- slug: tool-limitations
  similarity: 0.65
solutions:
- development-environment-optimization
- development-workflow-automation
- business-process-automation
- platform-independent-scripting-languages
- value-stream-mapping
- workaround-registry
- fast-feedback-loops
- self-service-developer-platform
- master-data-stewardship
- role-model-rationalization
layout: problem
lang: de
en_slug: increased-manual-work
---

## Description

Erhöhte manuelle Arbeit tritt auf, wenn Entwickler repetitive Routineaufgaben von Hand durchführen müssen, die durch Skripte, Werkzeuge oder Prozessverbesserungen automatisiert werden könnten. Dieser manuelle Overhead verringert die für kreative Problemlösung, Feature-Entwicklung und andere wertvolle Aktivitäten verfügbare Zeit. Häufige Beispiele umfassen manuelles Testen, Deployment-Prozesse, Dateneingabe, Dateimanipulation oder Umgebungs-Setup. Das Problem verschärft sich über die Zeit, während sich Teams an manuelle Prozesse gewöhnen und nicht in Automatisierung investieren.

## Indicators ⟡

- Entwickler führen dieselbe Abfolge von Schritten wiederholt für Routineaufgaben durch
- Erhebliche Zeit wird für Aufgaben aufgewendet, die sich mechanisch oder repetitiv anfühlen
- Fehler treten häufig in Routineprozessen aufgrund manueller Ausführung auf
- Teammitglieder äußern Frustration über für "Beschäftigungsarbeit" aufgewendete Zeit
- Ähnliche Aufgaben dauern viel länger als sie mit ordentlichem Tooling sollten

## Symptoms ▲

- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Zeit, die für repetitive manuelle Aufgaben aufgewendet wird, verringert direkt die für produktive Entwicklungsarbeit verfügbare Zeit.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Erhebliche Zeit für mühsame Beschäftigungsarbeit statt bedeutungsvolle Entwicklung aufzuwenden, führt zu Frustration und Entkopplung.
- [Inkonsistente Ausführung](inkonsistente-ausfuehrung.md)
<br/>  Manuelle Prozesse sind inhärent anfällig für Variation und produzieren inkonsistente Ergebnisse über Teammitglieder und Zeit hinweg.
- [Verringerte individuelle Produktivität](verringerte-individuelle-produktivitaet.md)
<br/>  Entwickler erledigen weniger bedeutungsvolle Arbeit, weil ein großer Teil ihrer Zeit in repetitive manuelle Aufgaben fließt.

## Causes ▼

- [Ineffiziente Prozesse](ineffiziente-prozesse.md)
<br/>  Schlechte Workflows, die nicht optimiert oder automatisiert wurden, schaffen unnötige manuelle Arbeit für Entwickler.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne automatisierte Tests müssen Entwickler Änderungen manuell verifizieren, was zu ihrer manuellen Arbeitslast beiträgt.
- [Manuelle Deployment-Prozesse](manuelle-deployment-prozesse.md)
<br/>  Nicht automatisierte Deployment-Prozesse sind eine wesentliche Quelle repetitiver manueller Arbeit für Entwicklungsteams.
- [Unzureichende Testinfrastruktur](unzureichende-testinfrastruktur.md)
<br/>  Fehlende Automatisierungswerkzeuge zwingen Entwickler, Test- und Verifikationsaufgaben manuell durchzuführen.

## Detection Methods ○

- **Zeittracking-Analyse:** Beobachtung, wie viel Zeit Entwickler mit repetitiven Aufgaben verbringen
- **Aufgabenhäufigkeitsanalyse:** Identifikation, welche manuellen Aufgaben am häufigsten durchgeführt werden
- **Fehlerraten-Tracking:** Messung von Fehlern in Routineprozessen, die automatisiert werden könnten
- **Entwicklerbefragungen:** Befragung von Teammitgliedern zu manuellen Aufgaben, die sie frustrieren
- **Prozessdokumentations-Review:** Analyse dokumentierter Prozesse zur Identifikation von Automatisierungsmöglichkeiten

## Examples

Ein Entwicklungsteam deployt Anwendungen manuell in die Produktion, indem es einer 47-Schritte-Checkliste folgt, die das Kopieren von Dateien, die Aktualisierung von Konfigurationseinstellungen, den Neustart von Diensten und die Ausführung von Datenbankmigrationen umfasst. Dieser Prozess dauert 3 Stunden und muss für jedes Release durchgeführt werden, was erhebliche Entwicklerzeit verbraucht und Gelegenheiten für Fehler schafft, wenn Schritte übersehen oder falsch durchgeführt werden. Ein weiteres Beispiel betrifft Entwickler, die manuell Testdaten erzeugen, indem sie Datenbankeinträge kopieren und modifizieren, wobei sie 30 Minuten vor jeder Testsitzung verbringen, um den angemessenen Datenzustand einzurichten, während dieser Prozess mit einem einfachen Skript automatisiert werden könnte, das konsistente Testumgebungen in Sekunden erzeugt.
