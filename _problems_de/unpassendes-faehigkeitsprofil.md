---
title: Unpassendes Fähigkeitsprofil
description: Teammitgliedern fehlt essenzielles Wissen oder Erfahrung, die für ihre
  zugewiesenen Rollen und Verantwortlichkeiten nötig sind.
category:
- Culture
- Management
- Team
related_problems:
- slug: skill-development-gaps
  similarity: 0.7
- slug: inexperienced-developers
  similarity: 0.65
- slug: insufficient-design-skills
  similarity: 0.65
- slug: inconsistent-knowledge-acquisition
  similarity: 0.65
- slug: uneven-workload-distribution
  similarity: 0.6
- slug: poor-teamwork
  similarity: 0.6
solutions:
- pair-and-mob-programming
- structured-onboarding-program
- technical-skills-development
- cross-functional-skill-development
- knowledge-rotation
- domain-experts
- internal-technical-coaching
- communities-of-practice
- code-reading-sessions
- technology-radar
layout: problem
lang: de
en_slug: inappropriate-skillset
---

## Description

Unpassendes Fähigkeitsprofil tritt auf, wenn Teammitgliedern Aufgaben oder Rollen zugewiesen werden, die Wissen, Erfahrung oder Fähigkeiten erfordern, die sie nicht besitzen. Diese Diskrepanz zwischen benötigten Fähigkeiten und tatsächlichen Kompetenzen führt zu verringerter Produktivität, erhöhten Fehlerraten und Frustration sowohl für die einzelne Person als auch für das Team. Das Problem kann aus schlechten Einstellungsentscheidungen, schnellen Technologiewechseln oder der Zuweisung von Teammitgliedern zu unvertrauten Domänen ohne ausreichende Vorbereitung entstehen.

## Indicators ⟡

- Teammitglieder bitten häufig um Hilfe bei grundlegenden Aufgaben im Zusammenhang mit ihrer Rolle
- Die Arbeitsqualität liegt durchgängig unter den Erwartungen für die zugewiesene Ebene
- Teammitglieder vermeiden bestimmte Arten von Aufgaben oder delegieren sie durchgängig an andere
- Schulungsbedarf ist für die Rolle erheblich höher als erwartet
- Der Fortschritt bei zugewiesener Arbeit ist viel langsamer als bei ähnlichen Aufgaben, die von Kollegen erledigt werden

## Symptoms ▲

- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Teammitglieder, die außerhalb ihrer Kompetenz arbeiten, führen aufgrund mangelnder Vertrautheit mit Best Practices mehr Defekte ein.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Entwickler, die mit unvertrauten Technologien oder Domänen kämpfen, brauchen erheblich länger, um Aufgaben abzuschließen.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Entwickler ohne ordentliche Fähigkeiten implementieren Workarounds statt ordentlicher Lösungen, weil sie den richtigen Ansatz nicht kennen.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Ständiges Kämpfen mit Aufgaben jenseits des eigenen Fähigkeitslevels führt zu Frustration und letztlich Burnout.
- [Wissensabhängigkeit](wissensabhaengigkeit.md)
<br/>  Teammitglieder mit Fähigkeitslücken bleiben abhängig von erfahrenen Kollegen für Anleitung und Entscheidungsfindung.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Teammitglieder, die außerhalb ihrer Kompetenz arbeiten, produzieren Code geringerer Qualität, weil ihnen Wissen über Best Practices fehlt.

## Causes ▼

- [Schlechte Planung](schlechte-planung.md)
<br/>  Schlechte Personalplanung weist Menschen Rollen zu, ohne zu bewerten, ob ihre Fähigkeiten den Anforderungen entsprechen.
- [Unzureichende Mentoring-Struktur](unzureichende-mentoring-struktur.md)
<br/>  Ohne strukturiertes Mentoring fehlt Teammitgliedern mit Fähigkeitslücken die Unterstützung, um benötigte Kompetenzen zu entwickeln.
- [Schnelles Teamwachstum](schnelles-teamwachstum.md)
<br/>  Schnelle Einstellung kann Fähigkeits-Matching beeinträchtigen, während Teams das Besetzen von Positionen über das Finden der richtigen Passung priorisieren.

## Detection Methods ○

- **Fähigkeitsbewertungs-Reviews:** Regelmäßige Bewertung der Fähigkeiten von Teammitgliedern gegenüber Rollenanforderungen
- **Analyse der Aufgabenerledigungszeit:** Vergleich der für Aufgaben aufgewendeten Zeit mit Branchen- oder Team-Benchmarks
- **Fehlerraten-Tracking:** Beobachtung von Defektraten und Korrelation mit individuellen Fähigkeitsniveaus
- **Analyse des Schulungsbedarfs:** Identifikation von Lücken zwischen aktuellen Fähigkeiten und Job-Anforderungen
- **Peer-Review-Feedback:** Sammlung von Input von Kollegen zu Leistung und Fähigkeiten von Teammitgliedern

## Examples

Ein Junior-Entwickler wird beauftragt, ein komplexes Microservices-System zu architektieren, obwohl er nur grundlegende Web-Entwicklungserfahrung hat. Er kämpft mit Konzepten verteilter Systeme, trifft schlechte Technologieentscheidungen und schafft eine Architektur mit erheblichen Skalierbarkeits- und Zuverlässigkeitsproblemen. Senior-Entwickler müssen ständig eingreifen, um Designprobleme zu beheben, und der Projektzeitplan verlängert sich um Monate, während der Junior-Entwickler Konzepte lernt, die er hätte kennen sollen, bevor er die Verantwortung übernahm. Ein weiteres Beispiel betrifft einen Datenbankadministrator, der mit traditionellen relationalen Datenbanken versiert ist und beauftragt wird, eine neue NoSQL-Datenplattform zu verwalten. Er wendet relationale Datenbankkonzepte unangemessen an, versäumt es, Abfragen für die NoSQL-Engine zu optimieren, und schafft Datenmodelle, die schlecht performen. Das System erlebt häufige Performance-Probleme und Dateninkonsistenzen, die die Einstellung externer Berater zur Lösung erfordern, was mehr kostet, als die anfängliche Einstellung eines angemessen qualifizierten Administrators gekostet hätte.
