---
title: Hohe Fluktuation
description: Neue Entwickler werden frustriert und verlassen das Team aufgrund schlechten
  Onboardings und Systemkomplexität.
category:
- Business
- Communication
- Process
related_problems:
- slug: difficult-developer-onboarding
  similarity: 0.75
- slug: team-churn-impact
  similarity: 0.7
- slug: new-hire-frustration
  similarity: 0.7
- slug: developer-frustration-and-burnout
  similarity: 0.65
- slug: increased-cognitive-load
  similarity: 0.65
- slug: overworked-teams
  similarity: 0.65
solutions:
- structured-onboarding-program
- knowledge-rotation
- sustainable-pace-practices
- psychological-safety-practices
- improvement-budget
- team-autonomy-and-empowerment
- knowledge-base
- team-retrospectives
- communities-of-practice
- internal-technical-coaching
layout: problem
lang: de
en_slug: high-turnover
---

## Description

Hohe Fluktuation tritt auf, wenn Entwickler das Team häufig verlassen, oft kurz nach dem Beitritt, aufgrund von Frustration über Systemkomplexität, schlechte Onboarding-Erfahrungen oder schwierige Arbeitsbedingungen. Dies schafft einen Teufelskreis, in dem die verbleibenden Teammitglieder ständig neue Leute einarbeiten müssen, statt sich auf Entwicklungsarbeit zu konzentrieren, während institutionelles Wissen kontinuierlich verloren geht. Hohe Fluktuation ist besonders schädlich für Legacy-Systeme, bei denen der Aufbau von Domänenwissen und Verständnis komplexer Codebasen erhebliche Zeit braucht.

## Indicators ⟡
- Neue Mitarbeiter verlassen das Unternehmen innerhalb ihrer ersten 6-12 Monate
- Austrittsgespräche erwähnen häufig Frustration über Codebasis-Komplexität oder fehlende Unterstützung
- Die Teamzusammensetzung ändert sich häufig, was es schwierig macht, konsistente Praktiken aufrechtzuerhalten
- Erhebliche Zeit wird für Rekrutierung und Interviews aufgewendet statt für Entwicklung
- Projekte verzögern sich, weil neue Teammitglieder umfangreiches Training benötigen

## Symptoms ▲

- [Wissenssilos](wissenssilos.md)
<br/>  Häufige Abgänge konzentrieren verbleibendes Wissen auf weniger Menschen, was gefährliche Einzelpunkte der Expertise schafft.
- [Implizites Wissen](implizites-wissen.md)
<br/>  Wenn erfahrene Entwickler ohne Wissenstransfer gehen, geht kritisches Systemverständnis verloren oder verbleibt nur bei verbliebenen Personen.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Ständiges Onboarding neuer Teammitglieder und der Verlust erfahrener Entwickler verringern die Gesamtproduktivität des Teams.
- [Mentoren-Burnout](mentoren-burnout.md)
<br/>  Verbleibende Senior-Entwickler werden erschöpft, weil sie kontinuierlich neue Mitarbeiter einarbeiten, die möglicherweise auch bald gehen.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Neue Entwickler, die mit dem System nicht vertraut sind, führen aufgrund fehlenden Domänenwissens und Systemverständnisses eher Fehler ein.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Ständige Rekrutierung, Onboarding und verlorene Produktivität durch häufige Abgänge erhöhen direkt die Gesamtentwicklungskosten.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Hohe Fluktuation bedeutet, dass erfahrene Entwickler gehen und durch weniger erfahrene ersetzt werden, was die Gesamtexpertise des Teams verringert.

## Causes ▼

- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Schlechte Onboarding-Erfahrungen frustrieren neue Mitarbeiter und lassen sie sich unterstützt fühlen, was zu frühen Abgängen beiträgt.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Die Arbeit mit schuldenbelastetem, komplexem Code ist demoralisierend für Entwickler, die qualitativ hochwertige Software schreiben wollen.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Unzureichende Dokumentation zwingt neue Entwickler, sich mühsam durch das Systemverständnis zu kämpfen, was Frustration und Burnout erhöht.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Überlastete Entwickler, die ständigem Feuerlöschen und Wartungslast gegenüberstehen, werden erschöpft und suchen nach besseren Möglichkeiten.

## Detection Methods ○
- **Fluktuationsraten-Tracking:** Beobachtung, wie lange neue Mitarbeiter bleiben, und Identifikation von Mustern bei Abgängen
- **Austrittsgespräch-Analyse:** Sammlung und Analyse von Feedback ausscheidender Entwickler
- **Time-to-Productivity-Metriken:** Nachverfolgung, wie lange neue Mitarbeiter brauchen, um effektive Mitwirkende zu werden
- **Onboarding-Zufriedenheitsumfragen:** Regelmäßiges Feedback von neuen Teammitgliedern zu ihrer Erfahrung
- **Rekrutierungskostenanalyse:** Nachverfolgung der Gesamtkosten für den ständigen Ersatz von Teammitgliedern

## Examples

Ein Finanzdienstleistungsunternehmen hat ein Legacy-Trading-System, das über 15 Jahre mit minimaler Dokumentation gebaut wurde. Von neuen Entwicklern wird erwartet, dass sie innerhalb von 30 Tagen produktiv werden, aber die Komplexität des Systems bedeutet, dass es typischerweise 6 Monate dauert, die Geschäftslogik und Codearchitektur zu verstehen. Die meisten neuen Mitarbeiter werden frustriert und verlassen das Unternehmen innerhalb von 4 Monaten, weil sie sich vom System überwältigt und vom Team nicht unterstützt fühlen. Die verbleibenden Senior-Entwickler sind so beschäftigt damit, neue Leute einzuarbeiten, dass sie keine Zeit haben, die Dokumentation zu verbessern oder das System zu vereinfachen, was den Kreislauf aufrechterhält. Über zwei Jahre hat das Team 12 Entwickler eingestellt, aber nur 3 gehalten, und mehr Zeit für Rekrutierung und Training aufgewendet als für tatsächliche Entwicklungsarbeit. Ein weiteres Beispiel betrifft eine Gesundheitsanwendung, bei der HIPAA-Compliance-Anforderungen zusätzliche Komplexität für neue Entwickler schaffen. Ohne ordentliches Training zu Gesundheitsvorschriften und sicheren Programmierpraktiken machen neue Entwickler Fehler, die umfangreiche Nacharbeit erfordern. Der Stress der Arbeit mit sensiblen Daten, kombiniert mit der Komplexität, sowohl das technische System als auch die regulatorischen Anforderungen zu lernen, führt dazu, dass viele Entwickler Positionen in weniger regulierten Branchen suchen.
