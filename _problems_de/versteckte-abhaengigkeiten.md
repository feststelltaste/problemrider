---
title: Versteckte Abhängigkeiten
description: Workarounds und Patches schaffen unerwartete Abhängigkeiten zwischen
  Systemkomponenten, die aus der Codestruktur nicht ersichtlich sind.
category:
- Architecture
- Code
related_problems:
- slug: unpredictable-system-behavior
  similarity: 0.8
- slug: hidden-side-effects
  similarity: 0.7
- slug: accumulation-of-workarounds
  similarity: 0.65
- slug: global-state-and-side-effects
  similarity: 0.65
- slug: tight-coupling-issues
  similarity: 0.65
- slug: circular-dependency-problems
  similarity: 0.65
solutions:
- modularization-and-bounded-contexts
- abstraction-layers
- feature-detection
- platform-independence
- platform-independent-time-zone-handling
- dependency-injection
- dependency-injection-container
- dependency-breaking-techniques
- change-impact-analysis
- workaround-registry
- application-portfolio-inventory
- duplication-detection
layout: problem
lang: de
en_slug: hidden-dependencies
---

## Description

Versteckte Abhängigkeiten entstehen, wenn Systemkomponenten auf Weisen voneinander abhängig werden, die aus ihren Schnittstellen, ihrer Dokumentation oder ihrer offensichtlichen Struktur nicht ersichtlich sind. Diese Abhängigkeiten entstehen oft durch Workarounds, gemeinsam genutzten globalen Zustand, implizite Timing-Annahmen oder Nebeneffekte, die nicht Teil des ursprünglichen Designs waren. Entwickler, die Änderungen an einer Komponente vornehmen, könnten unwissentlich Funktionalität in scheinbar unzusammenhängenden Teilen des Systems brechen, weil die tatsächlichen Abhängigkeiten nicht sichtbar oder dokumentiert sind.

## Indicators ⟡

- Änderungen in einem Modul brechen unerwartet Funktionalität in unzusammenhängenden Modulen
- Das Systemverhalten hängt auf nicht offensichtliche Weise von der Reihenfolge der Operationen ab
- Komponenten funktionieren isoliert korrekt, schlagen aber bei der Integration fehl
- Debugging zeigt Verbindungen zwischen Komponenten auf, die aus dem Code nicht ersichtlich waren
- Systemausfälle kaskadieren durch Komponenten, die eigentlich nicht zusammenhängen sollten

## Symptoms ▲

- [Unvorhersehbares Systemverhalten](unvorhersehbares-systemverhalten.md)
<br/>  Änderungen an einer Komponente brechen scheinbar unzusammenhängende Komponenten, weil die versteckte Abhängigkeit zwischen ihnen nicht sichtbar ist.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Modifikationen brechen unbeabsichtigt Funktionalität in Komponenten, die auf versteckten Annahmen oder undokumentierten Interaktionen beruhen.
- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Ein Ausfall in einer Komponente pflanzt sich durch versteckte Abhängigkeitsketten zu anderen fort, die nicht antizipiert wurden.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Die Rückverfolgung der Grundursache von Ausfällen ist extrem schwierig, wenn die tatsächliche Abhängigkeitskette unsichtbar ist.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Entwickler werden zurückhaltend, Code zu ändern, weil vergangene versteckte Abhängigkeiten unerwartete Ausfälle verursacht haben.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Entwickler brechen unwissentlich versteckte Abhängigkeiten mit Routineänderungen, was neue Fehler mit hoher Rate einführt.

## Causes ▼

- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Workarounds schaffen informelle Verbindungen zwischen Komponenten, die die vorgesehene Architektur umgehen und undokumentiert bleiben.
- [Globaler Zustand und Nebeneffekte](globaler-zustand-und-nebeneffekte.md)
<br/>  Gemeinsam genutzter globaler Zustand schafft implizite Kopplung zwischen Komponenten, die auf dieselben veränderbaren Daten zugreifen.
- [Schlechte Kapselung](schlechte-kapselung.md)
<br/>  Komponenten, die interne Implementierungsdetails offenlegen, erlauben anderen Komponenten, auf unerwartete Weise von diesen Details abhängig zu werden.
- [Informationsverfall](informationsverfall.md)
<br/>  Während Dokumentation veraltet, werden einst dokumentierte Abhängigkeiten für Entwickler unsichtbar.
- [Blindheit bei der Systemintegration](blindheit-bei-der-systemintegration.md)
<br/>  Versteckte Abhängigkeiten schaffen blinde Flecken bei der Systemintegration, was zu unerwarteten Ausfällen führt, wenn Komponenten interagieren.

## Detection Methods ○

- **Abhängigkeits-Mapping:** Dokumentation und Visualisierung tatsächlicher Laufzeitabhängigkeiten vs. offensichtlicher Designabhängigkeiten
- **Ausfallwirkungsanalyse:** Nachverfolgung, welche Komponenten betroffen sind, wenn bestimmte Komponenten ausfallen
- **Integrationstests:** Testen von Komponentenkombinationen, um versteckte gegenseitige Abhängigkeiten aufzudecken
- **Änderungsauswirkungsbewertung:** Beobachtung, welche Komponenten Modifikation erfordern, wenn sich andere ändern
- **Code-Analysewerkzeuge:** Nutzung statischer Analyse zur Identifikation potenzieller versteckter Verbindungen

## Examples

Ein Nutzerauthentifizierungsdienst hat einen Workaround, der Login-Versuche in eine temporäre Datei schreibt, um ein Datenbankverbindungsproblem zu umgehen. Das Reporting-Modul liest diese Datei heimlich, um Echtzeit-Nutzeraktivitätsberichte zu erzeugen, was eine versteckte Abhängigkeit schafft, die nirgendwo dokumentiert ist. Als das Authentifizierungsteam das Datenbankproblem behebt und die temporäre Datei entfernt, schlägt das Reporting-Modul auf mysteriöse Weise fehl. Ein weiteres Beispiel betrifft ein E-Commerce-System, bei dem das Bestandsmodul davon abhängt, dass das Warenkorb-Modul verlassene Warenkörbe innerhalb von 30 Minuten bereinigt, um Überverkauf zu verhindern, aber diese Abhängigkeit existiert nur als Kommentar in einer Konfigurationsdatei, die die meisten Entwickler nie sehen. Als der Warenkorb-Bereinigungsprozess so geändert wird, dass er alle 2 Stunden läuft, wird die Bestandsverfolgung ungenau, was zum Fehlschlagen von Kundenbestellungen führt.
