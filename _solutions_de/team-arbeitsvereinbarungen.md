---
title: Team-Arbeitsvereinbarungen
description: Explizitmachen der ungeschriebenen Erwartungen des Teams —
  wie Entscheidungen getroffen werden, wie mit Meinungsverschiedenheiten
  umgegangen wird, was geteilt wird, wofür ein Meeting da ist — und
  Überarbeitung, wenn sie scheitern.
category:
- Team
- Culture
- Process
problems:
- poor-teamwork
- team-dysfunction
- unclear-sharing-expectations
- nitpicking-culture
- style-arguments-in-code-reviews
- bikeshedding
- unproductive-meetings
- individual-recognition-culture
- inconsistent-execution
- poor-communication
- team-confusion
- language-barriers
- fear-of-conflict
- conflicting-reviewer-opinions
- author-frustration
- avoidance-behaviors
- blame-culture
- communication-risk-within-project
- inadequate-mentoring-structure
- inconsistent-onboarding-experience
- knowledge-sharing-breakdown
- mentor-burnout
- micromanagement-culture
- new-hire-frustration
- perfectionist-review-culture
- rapid-team-growth
- reduced-review-participation
- review-process-avoidance
- review-process-breakdown
- team-members-not-engaged-in-review-process
- unclear-documentation-ownership
- communication-breakdown
- insufficient-code-review
- merge-conflicts
- power-struggles
- team-coordination-issues
- extended-review-cycles
- inadequate-code-reviews
- inadequate-initial-reviews
- lack-of-ownership-and-accountability
layout: solution
lang: de
en_slug: team-working-agreements
related_solutions:
- slug: code-review-guidelines
  similarity: 0.7
- slug: psychological-safety-practices
  similarity: 0.7
- slug: written-first-communication
  similarity: 0.65
- slug: decision-rights-and-escalation
  similarity: 0.65
- slug: team-retrospectives
  similarity: 0.65
- slug: code-conventions
  similarity: 0.6
---

## Description

Eine Arbeitsvereinbarung ist eine kurze, explizite Aussage darüber, wie ein Team operiert: wie Entscheidungen getroffen werden, welche Reaktionszeiten Menschen voneinander erwarten können, wie Meinungsverschiedenheiten gelöst werden, was wo geteilt wird und welches Verhalten außerhalb der Grenzen liegt. Jedes Team hat diese Normen bereits; die Frage ist nur, ob sie ausgesprochen oder erschlossen werden. Unausgesprochene Normen versagen auf vorhersagbare Weisen — sie werden von Neuankömmlingen durch Fehler gelernt, sie unterscheiden sich zwischen Mitgliedern, die jeweils glauben, dass ihre offensichtlich sind, und man kann sich nicht auf sie berufen, wenn jemand sie verletzt, weil niemand auf etwas zeigen kann. Sie niederzuschreiben macht ein Team nicht funktionsfähig, und ein Team mit echten Konflikten wird diese nicht mit einem Dokument lösen. Was es tut, ist eine ganze Klasse von Reibung zu beseitigen, die daraus entsteht, dass Menschen auf inkompatiblen Annahmen operieren, während sie glauben, sie zu teilen.

## How to Apply ◆

> Langlebige Wartungsteams sammeln über Jahre ungeschriebene Konventionen an, und die resultierenden Erwartungen sind üblicherweise für jeden unsichtbar außer dem neuesten Mitglied, das sie ständig verletzt.

- **Schreiben Sie die Vereinbarung gemeinsam**, in einer ein- bis zweistündigen Sitzung. Eine von einem Lead herabgereichte Vereinbarung ist eine Richtlinie, und Richtlinien werden befolgt statt besessen. Die Diskussion ist mindestens so wertvoll wie das Artefakt, weil sie die Annahmen zutage bringt, von denen die Mitglieder nicht wussten, dass sie sich darüber uneinig waren.
- Beginnen Sie bei **echter Reibung, nicht bei einer Vorlage**. Fragen Sie, was kürzlich schiefgelaufen ist und was sich jede Person anders wünscht. Vereinbarungen, die aus generischen Best Practices zusammengesetzt sind, decken Situationen ab, die das Team nicht hat, und lassen die aus, die es hat.
- Decken Sie die Bereiche ab, die tatsächlich Konflikte erzeugen: **wie Entscheidungen getroffen werden und von wem, erwartete Reaktionszeiten für Reviews und Fragen, was wo niedergeschrieben wird, wie Meinungsverschiedenheiten eskaliert werden, Meeting-Normen und Verfügbarkeitserwartungen** über Zeitzonen oder Arbeitszeiten hinweg.
- Seien Sie **spezifisch genug, um überprüfbar zu sein**. "Wir kommunizieren offen" ist nicht falsifizierbar und daher nutzlos. "Reviews werden innerhalb eines Arbeitstages aufgenommen; wenn Sie das nicht können, sagen Sie es im Kanal" kann beobachtet, angerufen und verletzt werden.
- Adressieren Sie **Meinungsverschiedenheit explizit**. Geben Sie an, wie das Team damit umgeht — wer entscheidet, innerhalb welcher Zeit, und was mit der abweichenden Position geschieht. Teams, die Konflikte vermeiden, tun dies teilweise, weil sie keine vereinbarte Prozedur dafür haben, sodass jede Meinungsverschiedenheit droht, persönlich zu werden.
- Beziehen Sie **Normen für Review-Verhalten** ein, wenn Reviews ein Reibungspunkt sind: was einen Merge blockiert, was ein Vorschlag ist, und dass mechanisch überprüfbarer Stil die Aufgabe des Werkzeugs ist. Hier werden Nitpicking und Bikeshedding am günstigsten adressiert, weil beides üblicherweise die Abwesenheit eines vereinbarten Umfangs ist, statt eines Charakterzugs.
- Geben Sie für **verteilte oder mehrsprachige Teams** die Sprach- und Kanalkonventionen klar an: welche Sprache für Code, Kommentare, Tickets und Meetings genutzt wird; dass es erwartet statt unhöflich ist, jemanden zu bitten, etwas zu wiederholen oder umzuformulieren; und dass schriftliche Zusammenfassungen verbalen Entscheidungen folgen. Dies beseitigt eine große und selten diskutierte Quelle des Ausschlusses.
- **Halten Sie es auf einer Seite** und platzieren Sie es dort, wo neue Mitglieder es in ihrer ersten Woche antreffen. Eine Vereinbarung, die länger als eine Seite ist, wird nicht gelesen, und eine Vereinbarung, die nicht gelesen wird, ist schlimmer als keine, weil Menschen glauben, sie tue etwas.
- **Überarbeiten Sie sie, wenn sie scheitert.** Jede wiederkehrende Reibung ist eine Lücke in der Vereinbarung oder eine Regel, die nicht funktioniert. Eine ständige Retrospektiv-Frage — "haben wir unsere Vereinbarung befolgt, und wo hat sie nicht geholfen?" — hält sie am Leben; ohne dies wird sie zu einem Artefakt aus einem Onboarding-Ordner.

## Tradeoffs ⇄

> Explizite Vereinbarungen lösen die Reibung, die durch nicht übereinstimmende Annahmen verursacht wird, können aber echte Interessenkonflikte nicht lösen und können bei schlechter Formulierung zu einer bürokratischen Waffe werden.

**Vorteile:**

- Neue Mitglieder werden schneller effektiv, weil die Normen, die sie sonst durch Übertretung lernen würden, im Voraus angegeben sind.
- Wiederkehrende geringgradige Konflikte — Review-Nitpicking, Reaktionszeit-Erwartungen, wer was entscheidet — werden einmal geklärt, statt in jeder Instanz neu verhandelt zu werden.
- Problematisches Verhalten kann adressiert werden, indem auf eine gemeinsame Vereinbarung verwiesen wird, statt dass eine Person eine andere konfrontiert, was die persönlichen Kosten des Ansprechens dramatisch senkt.
- Meetings verbessern sich messbar, wenn ihr Zweck und ihre Normen angegeben sind, da die meisten unproduktiven Meetings unproduktiv sind, weil niemand vereinbart hat, wofür sie da sind.
- Verteilte und mehrsprachige Teams profitieren am meisten, da dies genau die Umgebungen sind, wo ungesagte Normen am weitesten auseinanderdriften und am schwersten zu erschließen sind.

**Kosten und Risiken:**

- Vereinbarungen verfallen zu ungelesenen Dokumenten, wenn sie nicht wieder aufgegriffen werden. Ein Team mit einer zwei Jahre alten Vereinbarung, die niemand angesehen hat, hat dasselbe Problem, mit dem es begonnen hat, plus ein falsches Gefühl, es adressiert zu haben.
- Sie können ein Team mit echten Vertrauensproblemen, einem missbräuchlichen Mitglied oder strukturellen Interessenkonflikten nicht reparieren, und der Versuch, sie so zu nutzen, verzögert die echte Intervention.
- Schriftliche Regeln können als Waffe eingesetzt werden. Ein Mitglied, das die Vereinbarung selektiv anruft, um Argumente zu gewinnen, verwandelt ein Koordinationswerkzeug in ein Compliance-Instrument.
- Die anfängliche Sitzung bringt Meinungsverschiedenheiten zutage, die zuvor unterdrückt waren, was der Punkt ist, aber unangenehm ist und genug psychologische Sicherheit erfordert, um produktiv statt schädlich zu sein.
- Übermäßig detaillierte Vereinbarungen werden zu Bürokratie und werden übelgenommen, besonders von erfahrenen Mitgliedern, die sie als Aussage lesen, dass ihnen nicht zugetraut wird, sich vernünftig zu verhalten.

## How It Could Be

Ein verteiltes Team von sieben Personen, das eine europäische Logistikplattform pflegte, erstreckte sich über vier Länder und drei Zeitzonen. Code-Reviews lagen drei bis vier Tage lang, Meetings liefen auf Englisch mit zwei Mitgliedern, die fast nichts beitrugen, und ein wiederkehrender Streit über das Commit-Message-Format hatte monatelang einen Teil jeder Retrospektive verbraucht. In einer zweistündigen Sitzung schrieben sie eine einseitige Vereinbarung: Reviews werden innerhalb eines Arbeitstages aufgenommen oder explizit im Kanal abgelehnt; in Meetings getroffene Entscheidungen werden innerhalb der Stunde schriftlich zusammengefasst; das Commit-Format wird durch einen Hook statt durch Menschen durchgesetzt; und eine explizite Aussage, dass das Bitten um eine Umformulierung in einem Meeting erwartet wird. Die Review-Latenz sank innerhalb von drei Wochen auf unter einen Tag. Die zwei stillen Mitglieder begannen, nach Meetings schriftlich beizutragen, was das Team nicht erwartet hatte, sich aber als ihr bevorzugter Modus herausstellte, und zwei bedeutende Design-Einwände kamen im ersten Monat auf diese Weise zutage.

Ein anderes Team nutzte seine Vereinbarung, um einen langwierigen Review-Konflikt zu beenden. Zwei leitende Entwickler hatten inkompatible Ansichten zur Fehlerbehandlung, und jeder Pull Request, der diesen Bereich berührte, wurde zu einer Pattsituation, die der Autor vermitteln musste. Die Vereinbarung fügte zwei Zeilen hinzu: mechanisch überprüfbare Regeln gehören in den Linter, und wo sich zwei Reviewer substanziell uneinig sind, entscheidet der Modul-Verantwortliche innerhalb eines Arbeitstages, und die abweichende Position wird im Pull Request protokolliert. Die erste Anwendung war unangenehm. Die vierte war Routine, und die protokollierten Abweichungen wurden später zum Input für ein Architecture Decision Record, das die zugrunde liegende Frage ordentlich klärte.
