---
title: Fehlausgerichtete Liefergegenstände
description: Das gelieferte Produkt oder Feature entspricht nicht den Erwartungen
  oder Anforderungen der Stakeholder.
category:
- Communication
- Process
related_problems:
- slug: stakeholder-developer-communication-gap
  similarity: 0.75
- slug: no-continuous-feedback-loop
  similarity: 0.7
- slug: communication-breakdown
  similarity: 0.65
- slug: product-direction-chaos
  similarity: 0.65
- slug: missed-deadlines
  similarity: 0.65
- slug: feedback-isolation
  similarity: 0.6
solutions:
- continuous-feedback
- stakeholder-feedback-loops
- on-site-customer
- personas
- prototypes
- prototyping
- requirements-traceability-matrix
- specification-by-example
- story-mapping
- subject-matter-reviews
- ubiquitous-language
- usability-tests
- user-acceptance-tests
- user-stories
- behavior-driven-development-bdd
- wireframing
layout: problem
lang: de
en_slug: misaligned-deliverables
---

## Description
Fehlausgerichtete Liefergegenstände sind ein klassisches Symptom eines Zusammenbruchs der Kommunikation zwischen einem Entwicklungsteam und seinen Stakeholdern. Dies tritt auf, wenn das finale Produkt nicht den Erwartungen des Geschäfts oder den Bedürfnissen der Nutzer entspricht. Diese Fehlausrichtung kann durch eine Vielzahl von Faktoren verursacht werden, von mehrdeutigen Anforderungen und fehlendem starkem Product Owner bis hin zum Versäumnis, Stakeholder während des gesamten Entwicklungsprozesses einzubeziehen. Das Ergebnis ist verschwendeter Aufwand, ein Produkt, das keinen Wert liefert, und ein Vertrauensverlust zwischen Team und Geschäft.

## Indicators ⟡
- Das Team muss Features ständig nacharbeiten, nachdem sie ausgeliefert wurden.
- Das Team erhält kein regelmäßiges Feedback von Stakeholdern.
- Das Team nutzt keinen Prototyp oder Mockup zur Klärung von Anforderungen.
- Das Team erhält während des Entwicklungsprozesses kein Feedback von Nutzern.

## Symptoms ▲

- [Unzufriedenheit der Stakeholder](unzufriedenheit-der-stakeholder.md)
<br/>  Wenn gelieferte Features nicht den Erwartungen entsprechen, verlieren Stakeholder das Vertrauen in das Entwicklungsteam.
- [Ressourcenverschwendung](ressourcenverschwendung.md)
<br/>  Fehlausgerichtete Liefergegenstände müssen umgebaut oder erheblich modifiziert werden, um tatsächlichen Anforderungen zu entsprechen, was Nacharbeit erzeugt.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Nacharbeitszyklen aufgrund fehlausgerichteter Liefergegenstände verzögern die Lieferung tatsächlichen Geschäftswerts an Nutzer.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Features, die nach falschen Spezifikationen gebaut wurden, stellen verschwendete Entwicklungszeit und -ressourcen dar.
- [Erosion des Nutzervertrauens](erosion-des-nutzervertrauens.md)
<br/>  Wiederholte Fehlausrichtung zwischen Geliefertem und Erwartetem untergräbt das Vertrauen zwischen Entwicklungs- und Geschäftsteams.

## Causes ▼

- [Kommunikationslücke zwischen Stakeholdern und Entwicklern](kommunikationsluecke-zwischen-stakeholdern-und-entwicklern.md)
<br/>  Schlechte Kommunikation zwischen Stakeholdern und Entwicklern führt zu Missverständnissen darüber, was gebaut werden muss.
- [Kein kontinuierlicher Feedback-Loop](kein-kontinuierlicher-feedback-loop.md)
<br/>  Ohne regelmäßiges Feedback während der Entwicklung wird Fehlausrichtung erst bei der Lieferung erkannt, wenn sie teuer zu beheben ist.
- [Annahmenbasierte Entwicklung](annahmenbasierte-entwicklung.md)
<br/>  Entwickler, die Annahmen über Anforderungen treffen statt sie zu validieren, führen zu Liefergegenständen, die Stakeholder-Erwartungen verfehlen.
- [Anforderungsmehrdeutigkeit](anforderungsmehrdeutigkeit.md)
<br/>  Mehrdeutige oder unvollständige Anforderungen lassen Raum für Fehlinterpretation und divergierende Erwartungen.

## Detection Methods ○

- **Regelmäßige Demos und Feedback-Sitzungen:** Häufige, iterative Demonstrationen funktionierender Software für Stakeholder, um frühes und kontinuierliches Feedback zu sammeln.
- **User Acceptance Testing (UAT):** Einbeziehung von Endnutzern oder Schlüssel-Stakeholdern beim Testen der Software, um sicherzustellen, dass sie ihre Bedürfnisse vor Release erfüllt.
- **Prototyping und Mockups:** Nutzung visueller Hilfsmittel früh im Prozess zur Validierung des Anforderungsverständnisses.
- **Klare Abnahmekriterien:** Sicherstellung, dass jede User Story oder Aufgabe gut definierte, messbare Abnahmekriterien hat, die sowohl von Entwicklern als auch Stakeholdern vereinbart wurden.
- **Post-Mortems/Retrospektiven:** Analyse von Projekten, bei denen Liefergegenstände fehlausgerichtet waren, zur Identifikation von Kommunikationszusammenbrüchen oder Prozessversagen.

## Examples
Ein Unternehmen investiert stark in ein neues internes Reporting-Werkzeug. Das Entwicklungsteam baut ein hochperformantes System mit komplexen Datenvisualisierungen. Bei der Veröffentlichung finden die Geschäftsnutzer es jedoch unbrauchbar, weil es eine einfache Export-nach-Excel-Funktion vermisst, die eine kritische, aber unausgesprochene Anforderung für ihren täglichen Arbeitsablauf war. In einem anderen Fall ist ein Mobile-App-Feature so gestaltet, dass Nutzer Fotos hochladen können. Die Entwickler implementieren eine grundlegende Upload-Funktion. Die Stakeholder erwarteten jedoch fortgeschrittene Bildbearbeitungsfunktionen (Zuschneiden, Filter), die nie explizit dokumentiert wurden, was zu einer erheblichen Lücke zwischen Erwartung und Lieferung führte. Dieses Problem ist ein klassisches Beispiel für ein Kommunikationsversagen in der Softwareentwicklung. Es ist besonders kostspielig, weil es oft zu erheblicher Nacharbeit und Verzögerungen führt, was sowohl Budget als auch Moral beeinträchtigt.
