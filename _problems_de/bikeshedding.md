---
title: Bikeshedding
description: Reviewer konzentrieren sich auf triviale Punkte wie Leerzeichen und
  Variablennamen, statt auf wichtigere Fragen wie Logik und Design.
category:
- Process
- Team
related_problems:
- slug: nitpicking-culture
  similarity: 0.65
- slug: gold-plating
  similarity: 0.55
- slug: unproductive-meetings
  similarity: 0.55
- slug: feature-creep
  similarity: 0.55
- slug: style-arguments-in-code-reviews
  similarity: 0.55
- slug: fear-of-conflict
  similarity: 0.55
solutions:
- psychological-safety-practices
- structured-communication-protocols
- code-review-guidelines
- team-working-agreements
- team-retrospectives
- decision-rights-and-escalation
- definition-of-done
- static-analysis-and-linting
- code-conventions
layout: problem
lang: de
en_slug: bikeshedding
---

## Description
Bikeshedding, auch bekannt als Parkinsons Gesetz der Trivialität, ist ein Phänomen, bei dem ein unverhältnismäßig großer Anteil an Zeit und Energie auf triviale und unbedeutende Details verwendet wird, während wichtigere und komplexere Fragen vernachlässigt werden. Dies tritt häufig in Meetings auf, in denen Teilnehmer schwierige Themen vermeiden und sich stattdessen auf leicht verständliche, aber letztlich unwichtige Details konzentrieren. Bikeshedding ist eine erhebliche Zeitverschwendung und kann ein Zeichen für eine dysfunktionale Teamkultur sein.

## Symptoms ▲

- [Oberflächliche Code-Reviews](oberflaechliche-code-reviews.md)
<br/>  Die Konzentration auf triviale Details verlängert Review-Zyklen, da Diskussionen sich in unwichtigen Kleinigkeiten verfangen.
- [Unproduktive Meetings](unproduktive-meetings.md)
<br/>  Meetings entarten zu Debatten über triviale Themen, während wichtige Design- und Logikfragen unadressiert bleiben.
- [Verpasste Termine](verpasste-termine.md)
<br/>  Zeit, die auf triviale Review-Diskussionen verschwendet wird, verzögert die Feature-Lieferung und verursacht Terminverschiebungen.
- [Störung der Entwicklung](stoerung-der-entwicklung.md)
<br/>  Übermäßiges Hin und Her bei trivialen Review-Kommentaren stört den Fokus und Arbeitsfluss der Entwickler.
- [Qualitäts-blinde Flecken](qualitaets-blinde-flecken.md)
<br/>  Während sich Reviewer auf Stil und Benennung fixieren, gehen kritische Logikfehler und Design-Mängel unentdeckt durch.

## Causes ▼

- [Unvollständiges Wissen](unvollstaendiges-wissen.md)
<br/>  Reviewer, denen ein tiefes Verständnis der Code-Logik fehlt, konzentrieren sich auf oberflächliche Fragen, die sie leicht bewerten können.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ohne automatisierte Stildurchsetzung werden triviale Formatierungsfragen zu Gegenständen manueller Review-Debatten.
- [Vermeidungsverhalten](vermeidungsverhalten.md)
<br/>  Reviewer vermeiden den kognitiven Aufwand der Bewertung komplexer Logik und konzentrieren sich stattdessen auf leicht zu beurteilende triviale Angelegenheiten.

## Detection Methods ○

- **Code-Review-Metriken:** Analyse der Arten von Kommentaren in Pull Requests (z. B. Verhältnis von stilistischen zu logischen/Design-Kommentaren).
- **Entwickler-Umfragen:** Befragung von Entwicklern zu ihrer Wahrnehmung der Code-Review-Wirksamkeit und häufigen Feedback-Arten.
- **Retrospektiven:** Diskussion von Code-Review-Prozessen und Identifikation wiederkehrender Frustrationen oder Ineffizienzen.
- **Reviewer-Schulung:** Beobachtung, ob Schulung zu wirksamen Code-Review-Praktiken die Qualität des Feedbacks verbessert.

## Examples

- **Szenario:** Ein Entwickler reicht einen Pull Request ein, der einen neuen, komplexen Algorithmus einführt. Die Review-Diskussion erstreckt sich über Tage, wobei 80 % der Kommentare debattieren, ob einfache oder doppelte Anführungszeichen für Strings verwendet werden sollen, während ein kritischer Grenzfall im Algorithmus unbemerkt bleibt.
- **Konkretes Beispiel:** Ein Team setzt ein neues Feature um, und während des Code-Reviews verbringt ein Senior-Entwickler eine Stunde damit, über die Namenskonvention für eine private Hilfsfunktion zu debattieren, obwohl das Projekt einen Linter hat, der solche Regeln automatisch durchsetzen könnte.
- **Kontext:** Dieses Problem tritt häufig auf, wenn Teams keine klaren Prozesse, automatisierten Werkzeuge oder ausreichenden Schulungen für Code-Reviews haben. Es kann die Entwicklungsgeschwindigkeit erheblich behindern und das Team davon abhalten, sich auf das zu konzentrieren, was für Codequalität und Projekterfolg wirklich zählt.
