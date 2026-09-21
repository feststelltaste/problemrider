---
title: Reviewer-Angst
description: Teammitglieder fühlen sich unsicher und ängstlich bei der Durchführung
  von Code-Reviews, was zu Vermeidung oder oberflächlichen Review-Praktiken führt.
category:
- Culture
- Process
- Team
related_problems:
- slug: fear-of-conflict
  similarity: 0.75
- slug: reduced-review-participation
  similarity: 0.75
- slug: author-frustration
  similarity: 0.75
- slug: reviewer-inexperience
  similarity: 0.75
- slug: review-process-avoidance
  similarity: 0.7
- slug: team-members-not-engaged-in-review-process
  similarity: 0.7
solutions:
- code-review-process-reform
- code-review-guidelines
- psychological-safety-practices
- pair-and-mob-programming
- checklists
- blameless-postmortems
- knowledge-rotation
- code-reading-sessions
- internal-technical-coaching
- team-retrospectives
layout: problem
lang: de
en_slug: reviewer-anxiety
---

## Description

Reviewer-Angst tritt auf, wenn Teammitglieder sich bei der Durchführung von Code-Reviews unsicher, eingeschüchtert oder ängstlich fühlen, oft aufgrund mangelnden Vertrauens in ihre Fähigkeiten, Angst, wichtige Probleme zu übersehen, oder Sorge, falsches Feedback zu geben. Diese Angst führt zu Review-Vermeidung, oberflächlichen Reviews, die sich nur auf offensichtliche Probleme konzentrieren, oder übermäßigem Zeitaufwand für Reviews aufgrund von Überanalyse und Selbstzweifeln.

## Indicators ⟡

- Teammitglieder melden sich freiwillig, um Code zu schreiben, vermeiden aber, den Code anderer zu reviewen
- Junior-Entwickler geben selten Review-Feedback zu Code von Senior-Entwicklern
- Reviews enthalten meist sichere, oberflächliche Kommentare statt substantiellem Feedback
- Reviewer verbringen übermäßig viel Zeit mit einfachen Änderungen aufgrund von Unsicherheit
- Teammitglieder äußern Unbehagen oder Stress bezüglich ihrer Review-Verantwortlichkeiten

## Symptoms ▲

- [Verringerte Review-Beteiligung](verringerte-review-beteiligung.md)
<br/>  Ängstliche Reviewer vermeiden es, sich freiwillig für Reviews zu melden, was den Pool aktiver Teilnehmer verringert.
- [Review-Engpässe](review-engpaesse.md)
<br/>  Wenn ängstliche Reviewer übermäßig lange für einfache Reviews brauchen oder Reviews ganz vermeiden, sinkt der Review-Durchsatz und schafft Engpässe.
- [Übereilte Genehmigungen](uebereilte-genehmigungen.md)
<br/>  Ängstliche Reviewer genehmigen Änderungen möglicherweise schnell, um das Unbehagen zu vermeiden, potenziell falsches oder kontroverses Feedback zu geben.
- [Zusammenbruch des Review-Prozesses](zusammenbruch-des-review-prozesses.md)
<br/>  Angst führt zu oberflächlichen Reviews, die sich auf sichere, oberflächliche Probleme konzentrieren, was die Gesamtwirksamkeit des Review-Prozesses untergräbt.

## Causes ▼

- [Unerfahrenheit der Reviewer](unerfahrenheit-der-reviewer.md)
<br/>  Mangel an Erfahrung und Fachwissen macht Reviewer unsicher über ihre Fähigkeit, wertvolles Feedback zu geben.
- [Schuldzuweisungskultur](schuldzuweisungskultur.md)
<br/>  In einer Schuldzuweisungskultur befürchten Reviewer, für die Genehmigung von Code verantwortlich gemacht zu werden, der später Probleme verursacht, was ihre Angst verstärkt.
- [Unzureichende Mentoring-Struktur](unzureichende-mentoring-struktur.md)
<br/>  Ohne Mentoring zum Aufbau von Review-Fähigkeiten und Vertrauen bleiben Teammitglieder ängstlich bezüglich ihrer Review-Fähigkeiten.
- [Angst vor Konflikt](angst-vor-konflikt.md)
<br/>  Ängstliche Reviewer entwickeln eine Angst vor Konfrontation und vermeiden herausforderndes Feedback, um ihr Unbehagen und potenzielle Konflikte zu minimieren.

## Detection Methods ○

- **Review-Beteiligungsanalyse:** Verfolgung, welche Teammitglieder aktiv an Code-Reviews teilnehmen
- **Review-Qualitätsbewertung:** Analyse der Tiefe und des Werts des von verschiedenen Reviewern gegebenen Feedbacks
- **Review-Zeitmuster:** Überwachung ungewöhnlich langer Review-Zeiten, die auf angstgetriebene Überanalyse hindeuten könnten
- **Teambefragungen:** Sammlung von Feedback zu Komfortniveau und Vertrauen beim Reviewen von Code
- **Feedback-Qualität bei Reviews:** Bewertung, ob Reviews wichtige Probleme erfassen oder sich nur auf oberflächliche Probleme konzentrieren

## Examples

Ein Junior-Entwickler im Team hat starke Programmierfähigkeiten, vermeidet aber konsequent das Reviewen der Pull Requests von Senior-Entwicklern, mit der Behauptung, „nicht qualifiziert" zu sein, die Arbeit erfahrenerer Kollegen zu reviewen. Wenn ihm Reviews zugewiesen werden, verbringt er Stunden mit der Analyse einfacher Änderungen und gibt nur sichere Kommentare zur Code-Formatierung ab, statt Logik oder Design zu prüfen. Seine Angst hindert ihn daran, wertvolle Perspektiven beizutragen, die den Code tatsächlich verbessern könnten. Ein weiteres Beispiel betrifft einen Entwickler mittlerer Erfahrungsstufe, der 2-3 Tage braucht, um Änderungen zu reviewen, die 30 Minuten dauern sollten, ständig sein Feedback anzweifelt und jeden Kommentar recherchiert, bevor er ihn postet. Sein Perfektionismus und seine Angst, falsch zu liegen, verursachen erhebliche Verzögerungen im Entwicklungsprozess, und er gibt oft übermäßig vorsichtiges Feedback, das die Codequalität nicht verbessert.
