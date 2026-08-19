---
title: Unklare Verantwortlichkeit für Dokumentation
description: Keine klare Verantwortung für die Pflege von Dokumentation führt zu
  veralteten, inkonsistenten oder fehlenden Informationen.
category:
- Communication
- Management
- Process
related_problems:
- slug: lack-of-ownership-and-accountability
  similarity: 0.8
- slug: poor-documentation
  similarity: 0.75
- slug: information-decay
  similarity: 0.65
- slug: information-fragmentation
  similarity: 0.65
- slug: master-data-ownership-gaps
  similarity: 0.6
- slug: legacy-system-documentation-archaeology
  similarity: 0.6
solutions:
- documentation-as-code
- living-documentation
- decision-rights-and-escalation
- clear-ownership-model
- knowledge-base
- team-working-agreements
- production-readiness-criteria
- application-portfolio-inventory
layout: problem
lang: de
en_slug: unclear-documentation-ownership
---

## Description

Unklare Verantwortlichkeit für Dokumentation tritt auf, wenn keine Einzelperson oder kein Team explizite Verantwortung für die Erstellung, Pflege und Aktualisierung der Systemdokumentation hat. Dies resultiert in Dokumentation, die veraltet, inkonsistent wird oder schlicht nicht existiert, weil alle annehmen, dass sich jemand anderes darum kümmert. Ohne klare Verantwortlichkeit wird Dokumentation zu einem sekundären Anliegen, das aufgeschoben wird, bis es zu einem kritischen Problem wird, zu welchem Zeitpunkt das Wissen, das zur Erstellung akkurater Dokumentation nötig ist, möglicherweise nicht mehr leicht verfügbar ist.

## Indicators ⟡

- Dokumentation existiert, aber niemand weiß, wer für ihre Aktualisierung verantwortlich ist
- Verschiedene Teammitglieder erstellen Dokumentation in unterschiedlichen Formaten und an unterschiedlichen Orten
- Dokumentationsaktualisierungen werden vergessen, wenn Systemänderungen vorgenommen werden
- Niemand überprüft Dokumentation auf Genauigkeit oder Vollständigkeit
- Dokumentationsverantwortlichkeiten sind nicht in Stellenbeschreibungen oder Leistungsbeurteilungen enthalten

## Symptoms ▲

- [Informationsverfall](informationsverfall.md)
<br/>  Ohne dass jemand dafür verantwortlich ist, Dokumentation aktuell zu halten, wird sie unvermeidlich über die Zeit veraltet und ungenau.
- [Informationsfragmentierung](informationsfragmentierung.md)
<br/>  Wenn niemand die Verantwortung für Dokumentation trägt, erstellen verschiedene Personen sie an unterschiedlichen Orten, was Wissen über mehrere Stellen verstreut.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Entwickler kämpfen damit, sich einzuarbeiten, wenn Dokumentation aufgrund unklarer Verantwortlichkeit veraltet, verstreut oder fehlend ist.
- [Wissenssilos](wissenssilos.md)
<br/>  Ohne gepflegte Dokumentation bleibt kritisches Wissen in den Köpfen einzelner Entwickler eingeschlossen, statt geteilt zu werden.
- [Doppelte Arbeit](doppelte-arbeit.md)
<br/>  Ohne gepflegte Dokumentation lösen Teammitglieder möglicherweise unwissentlich dieselben Probleme, die andere bereits angegangen sind.

## Causes ▼

- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Ein breiteres organisatorisches Muster unklarer Verantwortlichkeit erstreckt sich naturgemäß auf Dokumentationsverantwortlichkeiten.
- [Schlecht definierte Verantwortlichkeiten](schlecht-definierte-verantwortlichkeiten.md)
<br/>  Wenn Teamrollen und -verantwortlichkeiten nicht klar definiert sind, fällt die Dokumentationspflege durch die Ritzen.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter Terminendruck wird Dokumentation depriorisiert, und niemand wird für ihre Pflege zur Verantwortung gezogen.

## Detection Methods ○

- **Dokumentationsverantwortlichkeits-Audit:** Befragung von Teammitgliedern, wer ihrer Meinung nach für verschiedene Dokumentation verantwortlich ist
- **Aktualisierungshäufigkeitsanalyse:** Verfolgung, wie oft Dokumentation im Verhältnis zu Systemänderungen aktualisiert wird
- **Dokumentationsqualitätsbewertung:** Bewertung von Konsistenz und Genauigkeit bestehender Dokumentation
- **Verantwortlichkeitsmatrix-Überprüfung:** Analyse, ob Dokumentationsaufgaben klar zugewiesen sind
- **Verfolgung der Dokumentationsnutzung:** Überwachung, ob Teammitglieder tatsächlich bestehende Dokumentation nutzen

## Examples

Ein Entwicklungsteam hat umfassende API-Dokumentation, die während des initialen Systemdesigns erstellt wurde, aber niemand wurde damit beauftragt, sie zu pflegen. Über zwei Jahre haben sich die APIs erheblich weiterentwickelt mit neuen Endpunkten, geänderten Parametern und veralteter Funktionalität, aber die Dokumentation spiegelt immer noch das ursprüngliche Design wider. Neue Entwickler und Integrationspartner nutzen die veraltete Dokumentation und werden frustriert, wenn ihr Code nicht funktioniert. Jedes Teammitglied nimmt an, dass jemand anderes die Dokumentation aktualisieren wird, und die technischen Redakteure konzentrieren sich auf nutzerseitige statt Entwicklerdokumentation. Ein weiteres Beispiel betrifft ein System, bei dem verschiedene Entwickler Dokumentation in unterschiedlichen Wikis, gemeinsam genutzten Laufwerken und Code-Kommentaren erstellen, je nach ihren persönlichen Präferenzen. Wenn Teammitglieder Informationen benötigen, wissen sie nicht, wo sie suchen sollen, und verbringen oft mehr Zeit mit der Suche nach Dokumentation, als sie damit verbringen würden, das System direkt zu verstehen. Wichtige architektonische Entscheidungen sind in den persönlichen Notizen eines Entwicklers dokumentiert, was sie unzugänglich macht, wenn diese Person nicht verfügbar ist.
