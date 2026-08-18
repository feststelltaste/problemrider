---
title: Unzureichende Code-Reviews
description: Code-Reviews werden nicht durchgängig durchgeführt, sind übereilt, oberflächlich
  oder versäumen es, kritische Probleme zu identifizieren, was zu geringerer Codequalität
  und erhöhtem Risiko führt.
category:
- Code
- Process
related_problems:
- slug: insufficient-code-review
  similarity: 0.9
- slug: superficial-code-reviews
  similarity: 0.85
- slug: review-process-breakdown
  similarity: 0.8
- slug: code-review-inefficiency
  similarity: 0.8
- slug: inadequate-initial-reviews
  similarity: 0.8
- slug: team-members-not-engaged-in-review-process
  similarity: 0.8
solutions:
- code-review-process-reform
- checklists
- code-reviews
- secure-coding-guidelines
- secure-software-development
- security-policies-for-development
- static-code-analysis
- code-review-guidelines
- small-change-batches
- team-working-agreements
- work-in-progress-limits
layout: problem
lang: de
en_slug: inadequate-code-reviews
---

## Description
Unzureichende Code-Reviews sind ein wesentlicher Beitragender zu schlechter Softwarequalität. Dies umfasst sowohl oberflächliche Reviews, die wenig aussagekräftiges Feedback bieten, als auch inkonsistente Review-Praktiken. Wenn Code-Reviews übereilt, oberflächlich oder von unerfahrenen Reviewern durchgeführt werden, erfassen sie wahrscheinlich keine Fehler, Designfehler oder Abweichungen von Best Practices. Oberflächliche Reviews konzentrieren sich oft auf kleinere stilistische Fragen statt auf kritische Logik- oder Designfehler und bieten wenig mehr als "sieht für mich gut aus"-Genehmigungen ohne gründliche Untersuchung. Dies kann zu einer schrittweisen Verschlechterung der Codebasis führen, während technische Schulden und potenzielle Probleme sich anhäufen dürfen. Eine gesunde Code-Review-Kultur ist eine, in der Reviews gründlich, durchdacht und von einer vielfältigen Gruppe von Reviewern mit gemeinsamer Verantwortung für Codequalität durchgeführt werden.

## Indicators ⟡
- Code-Reviews sind oft ein Engpass im Entwicklungsprozess.
- Dieselben Arten von Fehlern werden wiederholt in Produktion gefunden.
- Entwickler lernen nicht voneinander durch Code-Reviews.
- Es gibt viel Debatte über Stil und andere triviale Fragen in Code-Reviews.

## Symptoms ▲

- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Wenn Code-Reviews es versäumen, Fehler und Designfehler zu erfassen, entkommen mehr Defekte in Produktionsumgebungen.
- [Inkonsistente Codebasis](inkonsistente-codebasis.md)
<br/>  Ohne gründliche Reviews, die Standards durchsetzen, verbreiten sich unterschiedliche Coding-Stile und -Muster unkontrolliert über die Codebasis.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Oberflächliche Reviews erlauben es, dass sich Abkürzungen und schlechte Designentscheidungen anhäufen, was technische Schulden über die Zeit erhöht.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Reviews, die Nebeneffekte und Kopplungsprobleme übersehen, erlauben Änderungen, die bestehende Funktionalität brechen.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Ohne aussagekräftiges Review-Feedback verschlechtert sich die Codequalität stetig, während schlechte Muster unwidersprochen bleiben.
- [Begrenztes Teamlernen](begrenztes-teamlernen.md)
<br/>  Oberflächliche Reviews eliminieren den Wissensaustausch-Vorteil von Code-Reviews, was Möglichkeiten für Teamlernen verringert.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Wenn Code-Reviews es versäumen, Probleme zu erfassen, steigt das Risiko, dass Fehler die Produktion erreichen, direkt an.
- [Unzureichende Erst-Reviews](unzureichende-erst-reviews.md)
<br/>  Wenn Code-Reviews generell unzureichend sind, werden Reviews der ersten Runde oberflächlich und übersehen kritische Probleme, die in späteren Runden erfasst werden müssen.

## Causes ▼

- [Zeitdruck](zeitdruck.md)
<br/>  Termindruck zwingt Reviewer, Reviews zu überstürzen, was zu oberflächlicher Untersuchung von Codeänderungen führt.
- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Übergroße Pull Requests überfordern Reviewer, was gründliche Untersuchung unpraktikabel macht und zu oberflächlichen Reviews führt.
- [Unerfahrenheit der Reviewer](unerfahrenheit-der-reviewer.md)
<br/>  Unerfahrenen Reviewern fehlt die Expertise, um tiefere Design-, Logik- oder Sicherheitsprobleme während Reviews zu identifizieren.
- [Überlastete Teams](ueberlastete-teams.md)
<br/>  Wenn Teams überlastet sind, werden Code-Reviews depriorisiert und überstürzt, um mit Lieferanforderungen Schritt zu halten.

## Detection Methods ○

- **Fehlerdichte nachverfolgen:** Eine hohe Anzahl an Fehlern in einem bestimmten Modul oder Feature kann darauf hindeuten, dass der Code nicht ordentlich reviewt wurde.
- **Code-Review-Kommentare analysieren:** Suche nach Mustern in den Kommentaren, um zu sehen, ob sich Reviewer auf die richtigen Dinge konzentrieren. Periodische Überprüfung einer Stichprobe von Code-Review-Kommentaren, um deren Tiefe und Fokus zu bewerten.
- **Post-Mortems/Retrospektiven:** Wenn Fehler in die Produktion gelangen, Analyse, ob sie im Code-Review hätten erfasst werden können und warum sie es nicht wurden.
- **Entwicklerbefragungen:** Befragung von Entwicklern zu ihrem Feedback zum Code-Review-Prozess und zur Qualität des Feedbacks, das sie während Reviews erhalten.
- **Codequalitätsmetriken:** Beobachtung von Metriken wie Fehlerdichte, technischen Schulden und Codekomplexität, die indirekt auf die Wirksamkeit von Reviews hindeuten können.
- **Nutzung statischer Analysewerkzeuge:** Diese Werkzeuge können automatisch viele verbreitete Probleme identifizieren, was Reviewern Zeit gibt, sich auf wichtigere Dinge zu konzentrieren.

## Examples
Ein Junior-Entwickler reicht einen Pull Request mit einem erheblichen Performance-Problem ein. Der Reviewer, der unter Termindruck steht, genehmigt den Pull Request, ohne das Problem zu bemerken. Das Performance-Problem wird später in Produktion entdeckt. Ein Entwickler reicht einen Pull Request ein, der einen N+1-Abfrage-Performance-Engpass einführt. Das Code-Review konzentriert sich ausschließlich darauf, ob die Variablennamen der Team-Konvention entsprechen und wo die geschweiften Klammern platziert sind, und übersieht das Performance-Problem völlig.

In einem anderen Fall hat ein Team eine Regel, dass alle Pull Requests von mindestens zwei Personen reviewt werden müssen. In der Praxis werden jedoch immer dieselben zwei Senior-Entwickler für die Reviews eingeteilt, und sie sind oft zu beschäftigt, um aussagekräftiges Feedback zu geben. Eine Sicherheitslücke wird in einem neuen Feature eingeführt, aber das Code-Review enthält nur Kommentare zur Code-Formatierung, und der Sicherheitsfehler wird erst viel später während eines Penetrationstests entdeckt. Dieses Problem ist verbreitet in Teams, die schnell wachsen, hohe Fluktuation haben oder unter Druck stehen, Features schnell zu liefern, oder wo die Bedeutung von Code-Reviews als Qualitätstor und Wissensaustausch-Mechanismus nicht vollständig verstanden wird.
