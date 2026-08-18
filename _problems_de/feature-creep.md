---
title: Feature-Creep
description: Der Umfang eines Features oder einer Komponente weitet sich im Laufe
  der Zeit schrittweise aus, was zu einem komplexen und überladenen System führt,
  das schwer zu warten ist.
category:
- Architecture
- Code
- Process
related_problems:
- slug: feature-creep-without-refactoring
  similarity: 0.85
- slug: scope-creep
  similarity: 0.75
- slug: uncontrolled-codebase-growth
  similarity: 0.7
- slug: large-feature-scope
  similarity: 0.7
- slug: feature-bloat
  similarity: 0.7
- slug: slow-feature-development
  similarity: 0.65
solutions:
- evolutionary-requirements-development
- formal-change-control-process
- product-owner
- requirements-analysis
- feature-toggles
- feature-usage-measurement
- definition-of-ready
- definition-of-done
- outcome-based-goal-setting
- regular-stakeholder-demonstrations
- customization-cost-attribution
- variant-consolidation
- explicit-extension-points
- fit-to-standard-principle
layout: problem
lang: de
en_slug: feature-creep
---

## Description
Feature-Creep ist die Tendenz, dass sich der Umfang eines Features oder einer Komponente im Laufe der Zeit ausweitet. Dies kann aus verschiedenen Gründen geschehen, etwa durch sich ändernde Anforderungen, mangelnden klaren Fokus oder den Wunsch, es allen recht zu machen. Feature-Creep kann zu einer Reihe von Problemen führen, darunter ein komplexes und überladenes System, das schwer zu warten ist, ein verwirrendes und überforderndes Nutzererlebnis sowie ein langer und unvorhersehbarer Entwicklungsprozess. Es ist ein verbreitetes Problem in der Softwareentwicklung und lässt sich nur schwer vermeiden.

## Indicators ⟡
- Das Team fügt dem System ständig neue Features hinzu.
- Das System wird im Laufe der Zeit immer komplexer.
- Die Benutzeroberfläche wird zunehmend überladen und verwirrend.
- Der Entwicklungsprozess wird länger und unvorhersehbarer.

## Symptoms ▲

- [Feature-Aufblähung](feature-aufblaehung.md)
<br/>  Unkontrollierter Feature-Creep führt direkt zu einem Produkt, das mit Features überladen ist, die seinen Kernwert verwässern.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Während das System durch angehäufte Features komplexer wird, dauert jede neue Ergänzung länger.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Die wachsende Komplexität durch Feature-Creep erhöht die Kosten für die Entwicklung, das Testen und die Wartung jedes zusätzlichen Features.
- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Sich ständig ausweitender Umfang verschiebt Liefertermine weiter nach hinten, während das Team versucht, mehr Features unterzubringen.
- [Scope Creep](scope-creep.md)
<br/>  Feature-Creep auf Komponentenebene trägt dazu bei, dass sich der Gesamtprojektumfang über die ursprünglichen Pläne hinaus ausweitet.
- [Nutzerverwirrung](nutzerverwirrung.md)
<br/>  Nutzer stoßen auf eine zunehmend komplexe Oberfläche mit zu vielen Optionen, was es schwerer macht, ihre Ziele zu erreichen.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Unkontrollierter Feature-Creep erhöht direkt die technischen Schulden, während das System komplexer wird, ohne dass die Architektur entsprechend angepasst wird.
- [Architektonische Fehlpassung](architektonische-fehlpassung.md)
<br/>  Die kontinuierliche Ergänzung von Features über den ursprünglichen Umfang hinaus kann das System über seine architektonische Auslegung hinaustreiben und eine wachsende Fehlpassung schaffen.

## Causes ▼

- [Häufige Anforderungsänderungen](haeufige-anforderungsaenderungen.md)
<br/>  Ständig wechselnde Anforderungen liefern einen stetigen Strom neuer Feature-Anfragen, die den Umfang ausweiten.
- [Übermäßige Gefälligkeit gegenüber Stakeholdern](uebermaessige-gefaelligkeit-gegenueber-stakeholdern.md)
<br/>  Teams, die jeder Stakeholder-Anfrage ohne Widerspruch zustimmen, lassen den Feature-Umfang kontinuierlich wachsen.
- [Kein formaler Änderungskontrollprozess](kein-formaler-aenderungskontrollprozess.md)
<br/>  Ohne formale Bewertung von Umfangsänderungen werden neue Features hinzugefügt, ohne ihre Auswirkung auf das Gesamtsystem zu prüfen.
- [Chaos in der Produktrichtung](chaos-in-der-produktrichtung.md)
<br/>  Fehlende klare Produktvision bedeutet, dass es keinen Rahmen gibt, um zu entscheiden, welche Features dazugehören und welche nicht.

## Detection Methods ○
- **Feature-Anfrage-Backlog:** Analyse des Feature-Anfrage-Backlogs, um Trends und Muster zu identifizieren.
- **Produkt-Roadmap:** Überprüfung der Produkt-Roadmap, ob sie fokussiert und realistisch ist.
- **Nutzerfeedback:** Anhören von Nutzerfeedback, um zu erkennen, ob das System als komplex und verwirrend empfunden wird.
- **Code-Komplexitätsmetriken:** Nutzung statischer Analysewerkzeuge, um die Komplexität der Codebasis zu messen.

## Examples
Ein Unternehmen entwickelt eine neue mobile App. Die App ist zunächst als einfache To-do-Listen-App konzipiert. Im Laufe der Zeit fügt das Team jedoch immer mehr Features hinzu. Sie ergänzen einen Kalender, eine Notizfunktion, eine Dateifreigabefunktion und eine Chat-Funktion. Die App wird so komplex, dass sie schwer zu bedienen ist, und das Team kann mit der Wartung nicht mehr Schritt halten. Das Unternehmen muss die App schließlich aufgeben und von vorn beginnen.
