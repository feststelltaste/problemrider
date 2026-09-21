---
title: Fehlende Eigenverantwortung und Rechenschaftspflicht
description: Keine klare Verantwortlichkeit für die Pflege von Codequalität, Dokumentation
  oder spezifischen Systemkomponenten über die Zeit.
category:
- Code
- Communication
- Process
related_problems:
- slug: unclear-documentation-ownership
  similarity: 0.8
- slug: delayed-issue-resolution
  similarity: 0.7
- slug: master-data-ownership-gaps
  similarity: 0.65
- slug: poorly-defined-responsibilities
  similarity: 0.65
- slug: poor-operational-concept
  similarity: 0.6
- slug: information-decay
  similarity: 0.6
solutions:
- clear-ownership-model
- clear-roles-and-ownership
- compatibility-governance
- decision-rights-and-escalation
- production-readiness-criteria
- team-retrospectives
- application-portfolio-inventory
- system-decommissioning
- team-working-agreements
- on-call-duty
- customization-under-version-control
- master-data-stewardship
- retention-and-disposal-policy
- role-model-rationalization
layout: problem
lang: de
en_slug: lack-of-ownership-and-accountability
---

## Description

Fehlende Eigenverantwortung und Rechenschaftspflicht tritt auf, wenn keine Einzelperson oder kein Team klare Verantwortung für die Pflege spezifischer Aspekte des Systems übernimmt, wie Codequalität, Dokumentation, Architekturentscheidungen oder Komponentenwartung. Dies führt zu einer "Tragödie der Allmende"-Situation, in der jeder annimmt, dass jemand anders wichtige, aber nicht dringende Aufgaben übernehmen wird. Ohne klare Eigenverantwortung werden kritische Wartungsaktivitäten aufgeschoben, Qualitätsstandards verschlechtern sich, und technische Schulden häufen sich an, bis Probleme kritisch werden.

## Indicators ⟡
- Wichtige Wartungsaufgaben werden durchgängig verzögert oder vergessen
- Niemand kann definitiv beantworten, wer für spezifische Systemkomponenten verantwortlich ist
- Kritische Dokumentation ist veraltet, weil sie niemand pflegt
- Qualitätsstandards variieren dramatisch über unterschiedliche Teile des Systems
- Probleme mit technischen Schulden werden identifiziert, aber nie priorisiert oder angegangen

## Symptoms ▲

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Ohne für Codequalität verantwortliche Eigentümer häufen sich technische Schulden an, weil niemand ihre Behebung priorisiert.
- [Informationsverfall](informationsverfall.md)
<br/>  Dokumentation verfällt, wenn niemand dafür verantwortlich ist, sie aktuell und akkurat zu halten.
- [Inkonsistente Qualität](inkonsistente-qualitaet.md)
<br/>  Die Qualität variiert dramatisch über das System, wenn niemand für die Wahrung von Standards verantwortlich ist.
- [Verzögerte Problemlösung](verzoegerte-problemloesung.md)
<br/>  Probleme bleiben ungelöst, weil niemand die Verantwortung übernimmt, sie anzugehen.
- [Qualitätsverschlechterung](qualitaetsverschlechterung.md)
<br/>  Die Systemqualität verschlechtert sich über die Zeit, wenn es keine Eigenverantwortung gibt, um sie zu pflegen und zu verbessern.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Ohne klare Eigenverantwortung wird Refactoring vermieden, weil sich niemand für die Verbesserung gemeinsam genutzten Codes verantwortlich fühlt.
- [Wartungslähmung](wartungslaehmung.md)
<br/>  Ohne klare Eigenverantwortung stockt die Wartungsarbeit, weil niemand die Verantwortung für Verbesserungen übernimmt.

## Causes ▼

- [Schlecht definierte Verantwortlichkeiten](schlecht-definierte-verantwortlichkeiten.md)
<br/>  Wenn Rollen und Verantwortlichkeiten nicht klar definiert sind, wird Eigenverantwortung natürlich mehrdeutig.
- [Auswirkung von Team-Fluktuation](auswirkung-von-team-fluktuation.md)
<br/>  Häufige Teamänderungen stören die Kontinuität der Eigenverantwortung, während ausscheidende Mitglieder unbetreute Komponenten zurücklassen.
- [Fehlpassung der Organisationsstruktur](fehlpassung-der-organisationsstruktur.md)
<br/>  Organisatorische Grenzen, die nicht mit der Systemarchitektur übereinstimmen, lassen Komponenten ohne klare Team-Eigenverantwortung zurück.
- [Vakuum an Projektautorität](vakuum-an-projektautoritaet.md)
<br/>  Das Fehlen klarer Projektautorität bedeutet, dass niemand Komponenten-Eigenverantwortung zuweist oder durchsetzt.

## Detection Methods ○
- **Verantwortlichkeits-Mapping:** Erstellung expliziter Matrizen, die zeigen, wer welche Komponenten und Qualitätsaspekte besitzt
- **Wartungsaufgaben-Tracking:** Überwachung, wie lange Wartungsaufgaben nicht zugewiesen oder unvollständig bleiben
- **Code-Review-Muster:** Beobachtung, ob bestimmten Codebereichen durchgängig gründliche Reviews fehlen
- **Dokumentations-Aktualität:** Nachverfolgung, wann unterschiedliche Dokumentationsabschnitte zuletzt aktualisiert wurden
- **Post-Incident-Analyse:** Untersuchung, ob Verzögerungen bei der Problemlösung aus unklarer Eigenverantwortung entstehen

## Examples

Eine große Webanwendung hat eine gemeinsam genutzte Authentifizierungsbibliothek, die ursprünglich von einem Entwickler entwickelt wurde, der das Unternehmen vor zwei Jahren verlassen hat. Seitdem wurden mehrere Sicherheitslücken in ähnlichen Bibliotheken gemeldet, aber niemand fühlt sich verantwortlich, den Authentifizierungscode zu auditieren oder zu aktualisieren. Unterschiedliche Teams nehmen an, dass "jemand in Security" oder "das Infrastrukturteam" sich darum kümmern wird, aber keines der beiden Teams betrachtet es als seine Verantwortung. Die Bibliothek wird weiterhin über Dutzende Anwendungen hinweg mit potenziellen Sicherheitsproblemen genutzt, weil niemand klare Rechenschaftspflicht für ihre Wartung hat. Ein weiteres Beispiel betrifft eine kritische Datenverarbeitungspipeline, bei der unterschiedliche Teams unterschiedliche Stufen gebaut haben. Wenn die Pipeline beginnt, falsche Ergebnisse zu produzieren, untersucht jedes Team seine eigene Stufe, aber niemand übernimmt Verantwortung für das Gesamtsystemverhalten. Das Problem besteht wochenlang, weil es Koordination zwischen Teams erfordert, aber keine einzelne Person oder Team besitzt den End-to-End-Prozess.
