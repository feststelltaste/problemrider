---
title: Schuldzuweisungskultur
description: Fehler werden bestraft statt konstruktiv angegangen, was Risikobereitschaft
  und Lernen entmutigt
category:
- Management
- Process
- Team
related_problems:
- slug: fear-of-failure
  similarity: 0.65
- slug: micromanagement-culture
  similarity: 0.65
- slug: workaround-culture
  similarity: 0.6
- slug: history-of-failed-changes
  similarity: 0.6
- slug: perfectionist-review-culture
  similarity: 0.6
- slug: perfectionist-culture
  similarity: 0.6
solutions:
- blameless-postmortems
- security-culture
- psychological-safety-practices
- team-working-agreements
- root-cause-analysis
- error-budgets
- team-autonomy-and-empowerment
- team-retrospectives
- defect-triage-process
- delivery-performance-metrics
layout: problem
lang: de
en_slug: blame-culture
---

## Description

Schuldzuweisungskultur besteht, wenn Organisationen auf Fehler, Ausfälle oder Probleme reagieren, indem sie sich darauf konzentrieren, die verantwortlichen Personen zu identifizieren und zu bestrafen, statt systemische Ursachen zu verstehen und Verbesserungen umzusetzen. Dies schafft ein Umfeld, in dem Teammitglieder risikoscheu werden, Probleme verbergen und die Übernahme von Verantwortung für Themen vermeiden. Die Kultur untergräbt Lernen, Innovation und wirksame Problemlösung, indem sie Menschen defensiv statt kooperativ macht, wenn Herausforderungen angegangen werden.

## Indicators ⟡

- Post-Incident-Diskussionen, die sich primär auf "wer" statt auf "was" und "warum" konzentrieren
- Teammitglieder werden defensiv oder ausweichend, wenn Probleme oder Ausfälle besprochen werden
- Zurückhaltung beim frühzeitigen Melden von Problemen, Beinaheunfällen oder potenziellen Problemen
- Individuelle Leistungsbeurteilungen, die Fehler stark gegenüber Lernen und Wachstum betonen
- Management-Sprache, die bei der Diskussion von Systemausfällen persönliche Schuld impliziert
- Teammitglieder vermeiden herausfordernde Aufgaben oder innovative Ansätze aufgrund des Ausfallrisikos
- Fehlende psychologische Sicherheit in Meetings, in denen Probleme besprochen werden

## Symptoms ▲

- [Angst vor Scheitern](angst-vor-scheitern.md)
<br/>  Wenn Fehler bestraft werden, entwickeln Teammitglieder eine durchdringende Angst davor, überhaupt einen Fehler zu machen, was Initiative erstickt.
- [Vermeidungsverhalten](vermeidungsverhalten.md)
<br/>  Teammitglieder vermeiden herausfordernde oder riskante Aufgaben, um ihre Exposition gegenüber potenzieller Schuld zu minimieren.
- [Verringerte Innovation](verringerte-innovation.md)
<br/>  Die Angst vor Schuld für gescheiterte Experimente tötet die Bereitschaft, neue Ansätze oder Technologien auszuprobieren.
- [Wissenssilos](wissenssilos.md)
<br/>  Menschen halten Informationen defensiv zurück, um sich selbst zu schützen, statt Wissen offen zu teilen.
- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Talentierte Entwickler verlassen Organisationen, in denen Schuldzuweisungskultur ein toxisches und stressiges Arbeitsumfeld schafft.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Entwickler vermeiden Refactoring, weil jede Regression aus Codeänderungen zu individueller Schuldzuweisung führen könnte.

## Causes ▼

- [Kultur der individuellen Anerkennung](kultur-der-individuellen-anerkennung.md)
<br/>  Belohnungssysteme, die auf individueller Leistung basieren, schaffen wettbewerbsorientierte Dynamiken, in denen die Fehler anderer vorteilhaft werden.

## Detection Methods ○

- Durchführung anonymer Umfragen zu psychologischer Sicherheit und Angst vor Konsequenzen
- Analyse von Vorfallreaktionsmustern zur Identifikation schuldfokussierter vs. lernfokussierter Diskussionen
- Beobachtung der Teilnahmeniveaus des Teams an Problemlösungsdiskussionen und Retrospektiven
- Überprüfung der in Vorfallberichten und Post-Mortem-Dokumentation verwendeten Sprache
- Befragung von Teammitgliedern zu ihrer Bereitschaft, Probleme zu melden oder neue Ansätze auszuprobieren
- Bewertung, ob systemische Verbesserungen aus Vorfallanalysen resultieren oder nur individuelle Maßnahmen
- Beobachtung von Teammoral, Stressniveaus und Fluktuationsraten
- Bewertung, ob Menschen von sich aus Informationen zu Problemen mitteilen oder direkt gefragt werden müssen

## Examples

Während eines größeren Produktionsausfalls schlägt ein Datenbankmigrationsskript fehl, weil es nicht ordentlich gegen das Produktionsdatenvolumen getestet wurde. Statt zu analysieren, warum der Testprozess dieses Problem nicht abgefangen hat, konzentriert sich das Management sofort auf den Entwickler, der das Skript geschrieben hat, kritisiert öffentlich sein Urteilsvermögen und führt zusätzliche Aufsicht für seine künftige Arbeit ein. Diese Reaktion sendet dem Team eine klare Botschaft, dass Einzelpersonen persönlich für Systemausfälle verantwortlich gemacht werden. Infolgedessen werden Entwickler extrem konservativ, verbringen übermäßig viel Zeit mit risikoarmen Aufgaben und vermeiden innovative Lösungen, die scheitern könnten. Als der nächste Vorfall auftritt – eine Sicherheitslücke, die mit besseren Code-Review-Prozessen hätte entdeckt werden können –, verbringt das Team das Post-Mortem-Meeting damit, defensiv ihre individuellen Handlungen zu erklären, statt kooperativ Systemverbesserungen zu identifizieren. Die Schuldzuweisungskultur verhindert, dass die Organisation lernt, dass beide Vorfälle Symptome unzureichender Prozesse waren, nicht individueller Inkompetenz.
