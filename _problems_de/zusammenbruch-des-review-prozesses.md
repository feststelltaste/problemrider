---
title: Zusammenbruch des Review-Prozesses
description: Code-Review-Praktiken versäumen es, kritische Probleme zu identifizieren,
  aussagekräftiges Feedback zu geben oder die Codequalität zu verbessern, aufgrund
  systemischer Prozessversagen.
category:
- Code
- Process
- Team
related_problems:
- slug: insufficient-code-review
  similarity: 0.8
- slug: inadequate-code-reviews
  similarity: 0.8
- slug: code-review-inefficiency
  similarity: 0.75
- slug: review-process-avoidance
  similarity: 0.75
- slug: team-members-not-engaged-in-review-process
  similarity: 0.75
- slug: superficial-code-reviews
  similarity: 0.75
solutions:
- code-review-process-reform
- code-review-guidelines
- team-working-agreements
- work-in-progress-limits
- checklists
- small-change-batches
- psychological-safety-practices
- pair-and-mob-programming
- team-retrospectives
- clear-roles-and-ownership
layout: problem
lang: de
en_slug: review-process-breakdown
---

## Description

Zusammenbruch des Review-Prozesses tritt auf, wenn Code-Review-Praktiken systematisch daran scheitern, ihre beabsichtigten Ziele der Verbesserung von Codequalität, Wissensaustausch und Fehlerprävention zu erreichen. Dies äußert sich als Reviews, die hastig, oberflächlich, inkonsistent sind oder ganz vermieden werden, was ein falsches Sicherheitsgefühl schafft, während sich Qualitätsprobleme in der Codebasis anhäufen. Der Zusammenbruch entsteht oft aus fehlausgerichteten Anreizen, Prozessreibung oder kulturellen Problemen, die effektives Review schwierig oder nicht lohnend machen.

## Indicators ⟡

- Code-Reviews übersehen konsequent offensichtliche Bugs oder Design-Mängel, die später in Produktion auftauchen
- Reviews konzentrieren sich primär auf Formatierung und Stil statt auf Logik, Architektur oder Wartbarkeit
- Große Änderungen werden mit minimaler Diskussion oder Feedback genehmigt
- Die Review-Durchlaufzeit ist entweder zu langsam (blockiert Entwicklung) oder zu schnell (deutet auf oberflächliches Review hin)
- Dieselben Arten von Problemen werden trotz Code-Review-Prozessen wiederholt in Produktion identifiziert

## Symptoms ▲

- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Wenn Reviews es versäumen, Fehler zu erfassen, erreichen mehr Bugs die Produktion, was die Fehlerrate erhöht.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ineffektive Reviews versäumen es, Coding-Standards durchzusetzen, was Inkonsistenzen in der gesamten Codebasis wuchern lässt.
- [Wissenssilos](wissenssilos.md)
<br/>  Zusammengebrochene Review-Prozesse beseitigen den Wissensaustauschvorteil von Reviews, sodass Expertise siloartig bleibt.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Reviews, die Code-Logik nicht gründlich prüfen, übersehen Regressionen, die zuvor funktionierende Funktionalität brechen.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Ohne effektive Reviews, die Design-Abkürzungen und schlechte Muster erfassen, häufen sich technische Schulden schneller in der Codebasis an.

## Causes ▼

- [Zeitdruck](zeitdruck.md)
<br/>  Zeitdruck begünstigt schnelle Genehmigungen gegenüber gründlichen Reviews, was die Review-Qualität im gesamten Team verschlechtert.
- [Unerfahrenheit der Reviewer](unerfahrenheit-der-reviewer.md)
<br/>  Unerfahrene Reviewer können tiefere Design-Probleme und Logikfehler nicht identifizieren, was die Review-Wirksamkeit einschränkt.
- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Überdimensionierte Pull Requests sind unmöglich gründlich zu reviewen, was Reviewer zu oberflächlicher Prüfung zwingt.
- [Verringerte Review-Beteiligung](verringerte-review-beteiligung.md)
<br/>  Wenn nur wenige Personen an Reviews teilnehmen, sind die verbleibenden Reviewer überlastet und können kein qualitatives Feedback geben.
- [Oberflächliche Code-Reviews](oberflaechliche-code-reviews.md)
<br/>  Oberflächliche Code-Reviews sind eine direkte Ursache und Form des Zusammenbruchs des Review-Prozesses — wenn sich Reviews nur auf Oberflächliches konzentrieren....

## Detection Methods ○

- **Review-Qualitätsanalyse:** Verfolgung, ob in Produktion gefundene Probleme im Review hätten erfasst werden können
- **Review-Beteiligungsmetriken:** Überwachung von Reviewer-Engagement, Feedback-Qualität und Diskussionstiefe
- **Review-Durchlaufzeit:** Messung der Zeit zwischen Review-Anfrage und aussagekräftigem Feedback
- **Post-Review-Bug-Verfolgung:** Analyse, ob der Review-Prozess Fehler effektiv verhindert
- **Bewertung des Wissenstransfers:** Bewertung, ob Reviews erfolgreich Wissen im Team austauschen
- **Review-Prozess-Befragungen:** Befragung von Teammitgliedern zu Review-Wirksamkeit und Schmerzpunkten

## Examples

Ein Entwicklungsteam hat Code-Review-Anforderungen eingeführt, aber Reviewer genehmigen konsequent große Pull Requests innerhalb von Minuten nach der Einreichung mit Kommentaren wie „LGTM", ohne Fragen zu stellen oder Feedback zu geben. Als Produktionsfehler auftreten, zeigt die Untersuchung, dass die Probleme jedem Reviewer offensichtlich gewesen wären, der die Code-Logik sorgfältig geprüft hätte. Das Team entdeckt, dass Reviewer Druck verspüren, schnell zu genehmigen, um Entwicklung nicht zu blockieren, und es gibt ein unausgesprochenes Verständnis, dass gründliches Review weniger wichtig ist als schnelle Genehmigung. Ein weiteres Beispiel betrifft ein Team, bei dem Code-Reviews zu Diskussionen über Code-Formatierung und Variablenbenennung ausarten, während bedeutende Design-Mängel, Sicherheitslücken und Performance-Probleme übersehen werden. Der Review-Prozess konzentriert sich auf subjektive Stilpräferenzen statt tatsächliche Probleme zu identifizieren, die Systemzuverlässigkeit und Wartbarkeit beeinträchtigen.
