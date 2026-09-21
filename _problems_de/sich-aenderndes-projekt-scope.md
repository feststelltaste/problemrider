---
title: Sich änderndes Projekt-Scope
description: Häufige Richtungswechsel im Projekt verwirren das Team und verhindern
  stetigen Fortschritt in Richtung Fertigstellung.
category:
- Management
- Process
related_problems:
- slug: scope-change-resistance
  similarity: 0.75
- slug: frequent-changes-to-requirements
  similarity: 0.75
- slug: no-formal-change-control-process
  similarity: 0.7
- slug: constantly-shifting-deadlines
  similarity: 0.7
- slug: scope-creep
  similarity: 0.7
- slug: unclear-goals-and-priorities
  similarity: 0.65
solutions:
- evolutionary-requirements-development
- formal-change-control-process
- product-owner
- explicit-prioritization-framework
- definition-of-ready
- regular-stakeholder-demonstrations
- story-mapping
- impact-mapping
- capacity-based-planning
layout: problem
lang: de
en_slug: changing-project-scope
---

## Description

Sich änderndes Projekt-Scope entsteht, wenn Projektanforderungen, -ziele oder -liefergegenstände während der Entwicklung häufig geändert werden, oft ohne ordentliche Bewertung der Auswirkung auf Zeitplan, Ressourcen oder Teammoral. Dies schafft Unsicherheit darüber, was das Team baut, stört das etablierte Entwicklungsmomentum und erzwingt ständige Neuplanung und Nacharbeit. Teams verlieren den Fokus und haben Schwierigkeiten, sinnvollen Fortschritt zu machen, wenn sich die Richtung häufig ändert.

## Indicators ⟡

- Projektanforderungen ändern sich mehrmals innerhalb kurzer Zeiträume
- Teammitglieder äußern Verwirrung über aktuelle Prioritäten und Ziele
- Zuvor abgeschlossene Arbeit wird aufgrund von Scope-Änderungen verworfen oder erheblich geändert
- Stakeholder liefern widersprüchliche oder sich entwickelnde Anforderungen
- Entwicklungsschätzungen werden aufgrund sich verschiebender Ziele unzuverlässig

## Symptoms ▲

- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Häufige Scope-Änderungen führen dazu, dass zuvor abgeschlossene Arbeit verworfen oder nachbearbeitet wird, was direkt Entwicklungsaufwand verschwendet.
- [Teamverwirrung](teamverwirrung.md)
<br/>  Ständig wechselnde Projektrichtung lässt Teammitglieder im Unklaren über aktuelle Ziele und Prioritäten.
- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Jede Scope-Änderung erfordert Neuplanung und Nacharbeit, was Liefertermine weiter nach hinten schiebt.
- [Demoralisierung des Teams](demoralisierung-des-teams.md)
<br/>  Das wiederholte Verwerfen abgeschlossener Arbeit aufgrund von Scope-Änderungen untergräbt die Teammotivation und das Vertrauen in die Führung.
- [Implementierungs-Nacharbeit](implementierungs-nacharbeit.md)
<br/>  Scope-Änderungen machen frühere Design-Entscheidungen ungültig und zwingen dazu, Features neu zu bauen, um neuen Anforderungen zu entsprechen.
- [Budgetüberschreitungen](budgetueberschreitungen.md)
<br/>  Unkontrollierte Scope-Änderungen erhöhen die insgesamt benötigte Arbeit, was dazu führt, dass Projekte ihr Budget überschreiten.
- [Ständig verschobene Termine](staendig-verschobene-termine.md)
<br/>  Sich änderndes Projekt-Scope verursacht direkt Terminverschiebungen, da das Team neue oder geänderte Anforderungen berücksichtigen muss.

## Causes ▼

- [Kein formaler Änderungskontrollprozess](kein-formaler-aenderungskontrollprozess.md)
<br/>  Ohne einen formalen Prozess zur Bewertung und Genehmigung von Änderungen erfolgen Scope-Änderungen ohne Auswirkungsbewertung.
- [Chaos in der Produktrichtung](chaos-in-der-produktrichtung.md)
<br/>  Widersprüchliche Stakeholder-Prioritäten und fehlende klare Produktführung führen dazu, dass sich die Projektrichtung wiederholt ändert.
- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Schlechte anfängliche Anforderungserhebung bedeutet, dass der tatsächliche Umfang schrittweise entdeckt wird, was wiederholte Änderungen erzwingt.

## Detection Methods ○

- **Analyse der Änderungsanfragehäufigkeit:** Nachverfolgung, wie oft und wie erheblich sich Anforderungen ändern
- **Bewertung der Auswirkung auf die Teamgeschwindigkeit:** Messung von Produktivitätseinbrüchen nach Scope-Änderungen
- **Stakeholder-Abstimmungsumfragen:** Bewertung, ob unterschiedliche Stakeholder ein konsistentes Verständnis der Ziele haben
- **Anforderungs-Traceability-Analyse:** Abbildung, wie sich Anforderungen im Laufe der Zeit entwickeln
- **Team-Moral-Monitoring:** Regelmäßige Check-ins zu Teamzufriedenheit und Klarheit

## Examples

Ein Projekt zur Entwicklung einer mobilen Anwendung beginnt mit dem Ziel, eine einfache App zur Ausgabenverfolgung zu erstellen. Zwei Wochen in die Entwicklung entscheiden Stakeholder, dass sie auch eine Belegscan-Funktionalität wollen. Einen Monat später wollen sie Budgetierungsfunktionen und Integration mit mehreren Banken hinzufügen. Jede Änderung erfordert erhebliche architektonische Modifikationen, und zuvor abgeschlossene Arbeit an der einfachen Ausgabenerfassung wird obsolet. Das Entwicklungsteam verbringt mehr Zeit mit dem Ändern bestehender Features als mit dem Bauen neuer, und der ursprüngliche Dreimonats-Zeitplan dehnt sich auf acht Monate aus. Ein weiteres Beispiel betrifft eine E-Commerce-Website, bei der sich die Geschäftsanforderungen wöchentlich basierend auf Konkurrenzanalysen ändern – zuerst wird ein bestimmter Checkout-Ablauf verlangt, dann völlig andere Zahlungsoptionen, dann eine gänzlich neue Produktkategorisierung. Entwickler stellen Features fertig, nur damit sie neu gestaltet werden, bevor sie deployt werden können, was zu Frustration und sinkendem Vertrauen in die Projektführung führt.
