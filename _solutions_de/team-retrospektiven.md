---
title: Team-Retrospektiven
description: Regelmäßige Überprüfung der Arbeitsweise des Teams und
  Änderung jeweils einer Sache, wobei die Änderungen wie jede andere Arbeit
  nachverfolgt werden.
category:
- Team
- Process
- Culture
problems:
- inefficient-processes
- poor-teamwork
- team-dysfunction
- resistance-to-change
- history-of-failed-changes
- inconsistent-execution
- bikeshedding
- workaround-culture
- limited-team-learning
- team-coordination-issues
- past-negative-experiences
- unclear-sharing-expectations
- team-confusion
- lack-of-ownership-and-accountability
- change-management-chaos
- code-review-inefficiency
- communication-breakdown
- decision-avoidance
- duplicated-work
- organizational-structure-mismatch
- overworked-teams
- poor-communication
- power-struggles
- reduced-code-submission-frequency
- reduced-team-productivity
- time-pressure
- accumulated-decision-debt
- author-frustration
- automated-tooling-ineffectiveness
- avoidance-behaviors
- blame-culture
- communication-risk-within-project
- delayed-decision-making
- fear-of-conflict
- fear-of-failure
- high-turnover
- increased-stress-and-burnout
- individual-recognition-culture
- mental-fatigue
- micromanagement-culture
- nitpicking-culture
- perfectionist-culture
- perfectionist-review-culture
- priority-thrashing
- process-design-flaws
- project-authority-vacuum
- reduced-review-participation
- review-bottlenecks
- review-process-avoidance
- review-process-breakdown
- reviewer-anxiety
- rushed-approvals
- team-demoralization
- team-members-not-engaged-in-review-process
- uneven-work-flow
- uneven-workload-distribution
- unmotivated-employees
- unproductive-meetings
layout: solution
lang: de
en_slug: team-retrospectives
related_solutions:
- slug: psychological-safety-practices
  similarity: 0.75
- slug: blameless-postmortems
  similarity: 0.7
- slug: delivery-performance-metrics
  similarity: 0.7
- slug: team-working-agreements
  similarity: 0.65
- slug: work-in-progress-limits
  similarity: 0.65
- slug: improvement-budget
  similarity: 0.65
---

## Description

Eine Retrospektive ist ein wiederkehrendes Meeting, in dem ein Team untersucht, wie es arbeitet, statt was es baut, und sich zu einer kleinen Anzahl konkreter Änderungen verpflichtet. Es ist der Mechanismus, durch den der Prozess eines Teams zu etwas wird, das das Team besitzt, statt zu etwas, das ihm auferlegt wird. Die Praxis ist weit verbreitet und weitgehend unwirksam, fast immer aus demselben Grund: Teams generieren Beobachtungen ohne Änderungen zu generieren, oder generieren Änderungen, die niemandem gehören und die nichts nachverfolgt. Eine Retrospektive, die eine Liste von Frustrationen und kein verändertes Verhalten produziert, lehrt das Team, dass das Ansprechen von Problemen sinnlos ist, was schlimmer ist als keine abzuhalten. In Legacy-Umgebungen verdient sich die Praxis ihre Kosten schnell, weil vieles von dem, was solche Arbeit schmerzhaft macht — der brüchige Deployment-Schritt, das Review, das immer wartet, das Modul, das jeder meidet — für das Management unsichtbar ist und nur das Team es weiß.

## How to Apply ◆

> Die größten Hindernisse eines Wartungsteams sind üblicherweise klein, spezifisch und langjährig, und sie bestehen fort, weil es keinen regelmäßigen Anlass gibt, bei dem jemand erwartet wird, sie zu benennen.

- Halten Sie sie in **fester Taktung** ab, statt wenn Dinge schiefgehen. Alle zwei Wochen ist typisch. Eine Retrospektive, die nur nach einer Krise abgehalten wird, wird mit Versagen assoziiert, was die ehrliche Berichterstattung entmutigt, von der sie abhängt.
- **Sammeln Sie Daten, bevor Sie sie interpretieren.** Beginnen Sie mit dem, was tatsächlich passiert ist — der Zeitlinie, den Zahlen, den Vorfällen — statt damit, wie sich Menschen dabei fühlen. Gefühle zählen und kommen als Nächstes, aber eine Diskussion, die mit Eindrücken beginnt, konvergiert auf denjenigen, der am eloquentesten ist.
- Produzieren Sie **höchstens zwei oder drei Maßnahmen**, jede mit einem benannten Verantwortlichen und einem Datum. Dies ist die einzige Änderung, die funktionierende Retrospektiven von nicht funktionierenden trennt. Eine Liste von zwölf Verbesserungen ist eine Liste von null Verbesserungen.
- Setzen Sie die Maßnahmen **auf dasselbe Board wie gewöhnliche Arbeit**, mit derselben Sichtbarkeit und derselben Erwartung des Abschlusses. Verbesserungsmaßnahmen, die in einem separaten Dokument gehalten werden, werden nicht geplant, nicht priorisiert und nicht erledigt.
- **Überprüfen Sie zuerst die Maßnahmen vom letzten Mal**, jedes Mal. Wenn sie nicht erledigt wurden, ist das das Wichtigste zu diskutieren — entweder fehlt dem Team die Kapazität, die Maßnahme war nicht wirklich vereinbart, oder etwas verhindert sie, und alle drei sind es wert, gewusst zu werden.
- **Rotieren Sie den Moderator.** Eine Retrospektive, die immer vom Team-Lead geführt wird, wird zu einem Statusmeeting, und es wird schwierig, irgendetwas über die Arbeitsweise des Leads anzusprechen.
- **Variieren Sie das Format** alle paar Sitzungen. Dieselben drei Fragen, die ein Jahr lang gestellt werden, produzieren dieselben drei Antworten. Wechseln Sie zwischen Zeitlinien-Reviews, fokussierten Tiefenanalysen zu einem wiederkehrenden Problem und vorausschauenden Formaten wie dem Vorstellen, wie das nächste Quartal scheitern könnte.
- **Eskalieren Sie, was das Team nicht beheben kann**, statt es wiederholt zu diskutieren. Manche Hindernisse sind organisatorisch, und eine Retrospektive, die monatlich zu ihnen zurückkehrt, ohne einen Weg nach außen, wird zu einem Ritual gemeinsamer Beschwerde. Leiten Sie sie explizit an denjenigen weiter, der handeln kann, und berichten Sie zurück.
- Halten Sie das Meeting **innerhalb einer festgelegten Zeitbox** und beenden Sie es mit den laut vorgelesenen Maßnahmen. Retrospektiven, die regelmäßig überziehen, werden als Kosten wahrgenommen, und diese Kostenwahrnehmung ist es, was sie während arbeitsreicher Zeiten abgesagt werden lässt — genau dann, wenn sie am meisten gebraucht werden.

## Tradeoffs ⇄

> Retrospektiven sind günstig und können sich zu erheblicher Verbesserung aufsummieren, erfordern aber psychologische Sicherheit, um ehrlich zu sein, und Nachverfolgung, um es wert zu sein, abgehalten zu werden.

**Vorteile:**

- Kleine, chronische Hindernisse werden behoben. Diese sind einzeln zu geringfügig, um eskaliert zu werden, machen aber zusammen einen großen Anteil der verlorenen Kapazität eines Wartungsteams aus.
- Prozessverbesserung wird kontinuierlich und vom Team besessen, statt als periodische Reorganisation von außen anzukommen.
- Probleme kommen zutage, während sie noch klein sind, da es einen geplanten und erwarteten Anlass gibt, sie anzusprechen.
- Das Team lernt aus seiner eigenen Geschichte, was der Weg ist, wie wiederkehrende Fehlermuster — die Migration, die immer bricht, die Schätzung, die immer falsch ist — schließlich adressiert statt wiederholt werden.
- Neuankömmlinge bekommen ein regelmäßiges, risikoarmes Forum, um Praktiken zu hinterfragen, die langjährige Mitglieder nicht mehr bemerken.

**Kosten und Risiken:**

- Ohne Nachverfolgung schadet die Praxis aktiv dem Vertrauen und lehrt das Team, dass das Ansprechen von Problemen Meetings statt Änderung produziert.
- Sie erfordern psychologische Sicherheit. In einer Schuldkultur produziert eine Retrospektive entweder Schweigen oder Schuldzuweisung, und die Retrospektive ist nicht die Intervention, die das behebt.
- Dieselben Beschwerden wiederholen sich, wenn die zugrunde liegenden Ursachen organisatorisch und außerhalb der Kontrolle des Teams sind, und das Meeting degradiert zu einer Beschwerdesitzung.
- Regelmäßige Meeting-Zeit ist eine echte Kosten, und sie ist das Erste, was gestrichen wird, wenn das Team unter Druck steht — genau dann, wenn die angesammelte Reibung am schlimmsten ist.
- Schlecht moderierte Retrospektiven können sich in Kritik an Einzelpersonen verwandeln, was dauerhaften Schaden anrichtet und schwer rückgängig zu machen ist.

## How It Could Be

Ein Team, das ein Lagerverwaltungssystem pflegte, hatte zwei Jahre lang Retrospektiven abgehalten und nach eigener Zählung über 200 Verbesserungsvorschläge generiert, von denen vier umgesetzt worden waren. Ihr neuer Lead änderte zwei Dinge: Das Meeting begann mit einer Überprüfung der vorherigen Maßnahmen, und nicht mehr als zwei Maßnahmen konnten übernommen werden, jede mit einem Verantwortlichen und einem Datum auf dem Team-Board. Die ersten zwei Monate waren unangenehm, weil die Antwort auf "haben wir es getan?" wiederholt Nein war. Im dritten Monat war es Ja. Im folgenden Jahr schlossen sie 21 von 24 zugesagten Maßnahmen ab, einschließlich der Reduzierung der Deployment-Prozedur von einer manuellen 40-Schritte-Checkliste zu einem Skript, das über die vorangegangenen zwei Jahre elfmal vorgeschlagen worden war, ohne dass es je jemand besaß.

Ein zweites Team nutzte ein fokussiertes Format, um ein wiederkehrendes Muster statt eines allgemeinen anzugehen. Ihre Releases waren dreimal hintereinander schlecht verlaufen, also verbrachten sie statt einer allgemeinen Retrospektive die gesamte Sitzung mit einer Zeitlinien-Rekonstruktion aller drei, nebeneinander gelegt. Der Vergleich machte sichtbar, was keine einzelne Post-Release-Diskussion gezeigt hatte: In allen drei Fällen war die Datenbankmigration in den letzten zwei Tagen geschrieben worden, von einer anderen Person als der, die die Anwendungsänderung geschrieben hatte. Die Maßnahme war eine einzige Regel — Migrationen werden mit der Änderung geschrieben und überprüft, die sie benötigt — und die nächsten sechs Releases verliefen ereignislos.
