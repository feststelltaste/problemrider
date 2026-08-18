---
title: Verlängerte Recherchezeit
description: Entwickler verbringen einen erheblichen Teil ihres Tages mit Recherche
  statt mit Umsetzung, aufgrund von Wissenslücken oder komplexen Legacy-Systemen.
category:
- Code
- Culture
- Process
related_problems:
- slug: duplicated-research-effort
  similarity: 0.7
- slug: difficult-developer-onboarding
  similarity: 0.7
- slug: knowledge-gaps
  similarity: 0.7
- slug: extended-cycle-times
  similarity: 0.7
- slug: increased-cognitive-load
  similarity: 0.65
- slug: inefficient-processes
  similarity: 0.65
solutions:
- knowledge-sharing-practices
- documentation-as-code
- knowledge-base
- architecture-documentation
- technical-spike
- knowledge-rotation
- living-documentation
- code-reading-sessions
- communities-of-practice
- internal-technical-coaching
- written-first-communication
layout: problem
lang: de
en_slug: extended-research-time
---

## Description

Verlängerte Recherchezeit entsteht, wenn Entwickler unverhältnismäßig viel Arbeitszeit mit Recherchieren, Untersuchen und Verstehen von Systemen, Anforderungen oder technischen Ansätzen verbringen müssen, statt aktiv Lösungen umzusetzen. Dieser Rechercheaufwand verringert erheblich die produktive Entwicklungszeit und deutet oft auf zugrunde liegende Probleme mit Systemkomplexität, Dokumentationsqualität oder Wissensverteilung im Team hin. Während etwas Recherche normal und wertvoll ist, wird verlängerte Recherchezeit problematisch, wenn sie durchgängig die Entwicklungsarbeit dominiert.

## Indicators ⟡

- Entwickler verbringen mehr als 30 % ihrer Zeit mit Recherche statt mit Programmieren
- Einfache Aufgaben erfordern umfangreiche Untersuchung, bevor mit der Umsetzung begonnen werden kann
- Teammitglieder werden häufig blockiert, während sie auf Informationen oder Verständnis warten
- Rechercephasen von Projekten dauern durchgängig länger als geschätzt
- Ähnliche Rechercheanfragen werden wiederholt von unterschiedlichen Teammitgliedern gestellt

## Symptoms ▲

- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Wenn Entwickler die meiste Zeit mit Recherche statt mit Programmieren verbringen, sinkt die Lieferrate des Teams erheblich.
- [Verlängerte Durchlaufzeiten](verlaengerte-durchlaufzeiten.md)
<br/>  Der Rechercheaufwand fügt dem Gesamtzyklus erhebliche Zeit hinzu, da Aufgaben viel länger dauern als die tatsächliche Umsetzungsarbeit.
- [Verringerte individuelle Produktivität](verringerte-individuelle-produktivitaet.md)
<br/>  Entwickler erledigen weniger Aufgaben, wenn ein unverhältnismäßiger Teil ihrer Zeit von Rechercheaktivitäten verbraucht wird.
- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Durchgängig unterschätzte Rechercephasen führen dazu, dass Projekte länger dauern als geplant.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Den Großteil des Tages mit Recherche statt mit Bauen zu verbringen, kann demoralisierend sein, besonders wenn dieselben Fragen wiederkehren.
- [Große Schätzungen für kleine Änderungen](grosse-schaetzungen-fuer-kleine-aenderungen.md)
<br/>  Entwickler geben große Schätzungen selbst für scheinbar einfache Änderungen ab, weil sie wissen, dass zunächst erhebliche Recherche nötig sein wird.

## Causes ▼

- [Wissenslücken](wissensluecken.md)
<br/>  Fehlendes Verständnis des Systems, der Domäne oder der Technologie zwingt Entwickler dazu, umfangreiche Zeit mit Recherche zu verbringen, bevor sie umsetzen können.
- [Informationsverfall](informationsverfall.md)
<br/>  Veraltete oder unvollständige Dokumentation zwingt Entwickler dazu, Systemverhalten von Grund auf zu recherchieren, statt sich auf bestehende Dokumentation zu verlassen.
- [Implizites Wissen](implizites-wissen.md)
<br/>  Wenn kritisches Systemwissen nur als stillschweigendes Wissen existiert, statt dokumentiert zu sein, müssen Entwickler Zeit aufwenden, um es durch Recherche zu entdecken.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Schwer verständlicher Code erfordert umfangreiche Untersuchung, bevor Entwickler sicher Änderungen vornehmen können.
- [Dokumentations-Archäologie bei Legacy-Systemen](dokumentations-archaeologie-bei-legacy-systemen.md)
<br/>  Wenn Systemwissen nur in veralteten Formaten und den Erinnerungen ausgeschiedener Mitarbeiter existiert, ist umfangreiche Recherche für jede Änderung nötig.

## Detection Methods ○

- **Zeittracking-Analyse:** Beobachtung des Prozentsatzes der Zeit, die für Recherche vs. Umsetzungsaktivitäten aufgewendet wird
- **Aufgabenaufteilungsanalyse:** Vergleich von Recherchezeit-Schätzungen mit tatsächlich aufgewendeter Zeit
- **Wissensaudit:** Identifikation wiederkehrender Recherchethemen, die auf systematische Wissenslücken hindeuten
- **Fragemuster-Analyse:** Nachverfolgung wiederholter Fragen, die auf fehlende Dokumentation oder fehlendes Wissen hindeuten
- **Entwickler-Umfragen:** Befragung von Teammitgliedern zu Hindernissen für effiziente Umsetzung

## Examples

Ein Entwicklungsteam, das an einer Gesundheitsanwendung arbeitet, verbringt 60 % seiner Zeit mit der Recherche von HIPAA-Compliance-Anforderungen, medizinischer Terminologie und Prozessen des Gesundheitsworkflows, weil die ursprünglichen Systemarchitekten und Fachexperten das Unternehmen verlassen haben. Jedes neue Feature erfordert Tage der Recherche zu regulatorischen Anforderungen und klinischen Workflows, bevor Code geschrieben werden kann. Ein weiteres Beispiel betrifft ein Team, das ein Machine-Learning-System wartet, bei dem Entwickler umfangreiche Zeit damit verbringen müssen, algorithmische Ansätze zu recherchieren, komplexe Datenpipelines zu verstehen und Performance-Optimierungstechniken zu untersuchen, weil die ursprünglichen Implementierer hochmoderne Techniken nutzten, die schlecht dokumentiert und vom aktuellen Team nicht gut verstanden sind.
