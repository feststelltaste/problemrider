---
title: Teammitglieder nicht in den Review-Prozess eingebunden
description: Code-Reviews werden oft denselben Personen zugewiesen, oder Reviewer
  geben kein aussagekräftiges Feedback, was zu einem Engpass und verringerter
  Qualität führt.
category:
- Communication
- Process
related_problems:
- slug: reduced-review-participation
  similarity: 0.85
- slug: inadequate-code-reviews
  similarity: 0.8
- slug: review-process-breakdown
  similarity: 0.75
- slug: reviewer-inexperience
  similarity: 0.75
- slug: insufficient-code-review
  similarity: 0.75
- slug: code-review-inefficiency
  similarity: 0.75
solutions:
- code-review-process-reform
- code-review-guidelines
- team-working-agreements
- small-change-batches
- psychological-safety-practices
- pair-and-mob-programming
- work-in-progress-limits
- team-retrospectives
- code-reading-sessions
- communities-of-practice
layout: problem
lang: de
en_slug: team-members-not-engaged-in-review-process
---

## Description
Wenn Teammitglieder vom Code-Review-Prozess losgelöst sind, hört er auf, ein effektives Werkzeug für Qualitätssicherung und Wissensaustausch zu sein. Dieses Problem äußert sich als Reviewer, die Gefälligkeitsgenehmigungen ohne sorgfältige Prüfung geben, oder eine kleine, überlastete Teilmenge des Teams, die alle Reviews durchführt. Dieser Mangel an Engagement kann zu einem Rückgang der Codequalität, der Verbreitung schlechter Praktiken und einer verpassten Gelegenheit für Mentoring und kollektive Code-Eigentümerschaft führen. Die Förderung einer Kultur, in der sich jeder für die Qualität der Codebasis verantwortlich fühlt, ist essentiell für ein gesundes Entwicklungsteam.

## Indicators ⟡
- Immer dieselben Personen werden mit dem Reviewen von Code betraut.
- Reviewer geben kein aussagekräftiges Feedback.
- Code-Reviews sind ein Engpass im Entwicklungsprozess.
- Das Team hat keine Kultur gemeinsamer Code-Eigentümerschaft.

## Symptoms ▲

- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Ohne aussagekräftiges Code-Review-Feedback gelangen schlechte Designentscheidungen und schlechte Praktiken unkontrolliert in die Codebasis.
- [Review-Engpässe](review-engpaesse.md)
<br/>  Wenn nur wenige Teammitglieder aktiv Code reviewen, stauen sich Pull Requests, während sie auf deren Aufmerksamkeit warten.
- [Wissenssilos](wissenssilos.md)
<br/>  Wenn nur wenige Personen Code reviewen, bleibt Wissen über die Codebasis konzentriert statt im Team verteilt.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Review-Engpässe durch desengagierte Reviewer verzögern die Integration und Veröffentlichung abgeschlossener Arbeit.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Gefälligkeitsgenehmigungen ohne aussagekräftige Prüfung resultieren in effektiv unzureichender Code-Review-Abdeckung.

## Causes ▼

- [Team-Silos](team-silos.md)
<br/>  Wenn Entwickler isoliert arbeiten, fühlen sie sich von Code außerhalb ihres Bereichs abgekoppelt und haben keine Motivation, ihn zu reviewen.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ohne klare Review-Standards und Erwartungen wissen Teammitglieder nicht, was ein gutes Review ausmacht, und greifen standardmäßig zu Gefälligkeitsgenehmigungen.
- [Team-Dysfunktion](team-dysfunktion.md)
<br/>  Zwischenmenschliche Probleme und mangelnde Kultur gemeinsamer Code-Eigentümerschaft verhindern aussagekräftiges Engagement im Review-Prozess.

## Detection Methods ○

- **Code-Review-Metriken:** Verfolgung von Metriken wie Review-Durchlaufzeit, Anzahl der Kommentare pro Review und Verteilung der Reviews unter Teammitgliedern.
- **Team-Befragungen/Interviews:** Befragung von Teammitgliedern zu ihrer Wahrnehmung des Code-Review-Prozesses, der Arbeitsbelastung und der Wirksamkeit.
- **Retrospektiven:** Diskussion von Code-Review-Herausforderungen und Identifikation wiederkehrender Muster von Desengagement.
- **Beobachtung:** Beobachtung der Teamdynamik während Stand-ups oder Diskussionen über Pull Requests.

## Examples
Ein Team hat eine Richtlinie, dass jeder Pull Request zwei Genehmigungen benötigt. Jedoch reviewen nur zwei Senior-Entwickler konsequent Code. Dies schafft einen Engpass, und Pull Requests warten oft tagelang auf Review, was Releases verzögert. In einem anderen Fall reicht ein Junior-Entwickler einen Pull Request ein, und der zugewiesene Reviewer genehmigt ihn einfach ohne Kommentare, obwohl es mehrere klare Verbesserungsmöglichkeiten im Design und der Testabdeckung des Codes gibt. Dieses Problem deutet oft auf zugrunde liegende Probleme in der Teamkultur, dem Arbeitslastmanagement oder der Prozessdefinition hin. Ein engagierter Code-Review-Prozess ist entscheidend für die Aufrechterhaltung der Codequalität, die Förderung des Wissensaustauschs und den Aufbau eines zusammenhängenden Teams.
