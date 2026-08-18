---
title: Negative Erfahrungen aus der Vergangenheit
description: Eine Situation, in der Entwickler zögern, Änderungen an der Codebasis
  vorzunehmen, aufgrund negativer Erfahrungen in der Vergangenheit.
category:
- Process
- Team
related_problems:
- slug: history-of-failed-changes
  similarity: 0.65
- slug: fear-of-breaking-changes
  similarity: 0.6
- slug: fear-of-change
  similarity: 0.6
- slug: resistance-to-change
  similarity: 0.55
- slug: brittle-codebase
  similarity: 0.55
- slug: maintenance-paralysis
  similarity: 0.55
solutions:
- blameless-postmortems
- psychological-safety-practices
- small-change-batches
- rollback-mechanisms
- automated-tests
- feature-flags
- mikado-method
- team-retrospectives
- pilot-projects
layout: problem
lang: de
en_slug: past-negative-experiences
---

## Description
Negative Erfahrungen aus der Vergangenheit sind eine Situation, in der Entwickler zögern, Änderungen an der Codebasis vorzunehmen, aufgrund negativer Erfahrungen in der Vergangenheit. Dies ist ein häufiges Problem in Teams mit einer brüchigen Codebasis und fehlenden automatisierten Tests. Negative Erfahrungen aus der Vergangenheit können zu einer Reihe von Problemen führen, einschließlich Angst vor Veränderung, einer Verlangsamung der Entwicklungsgeschwindigkeit und einem allgemeinen Gefühl der Stagnation.

## Indicators ⟡
- Entwickler zögern, Änderungen an der Codebasis vorzunehmen.
- Das Team ist nicht bereit, Risiken einzugehen.
- Das Team innoviert nicht.
- Das Team lernt nicht aus seinen Fehlern.

## Symptoms ▲

- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Entwickler, die Produktionsausfälle oder Schuldzuweisungen durch vergangene Änderungen erlebt haben, werden zurückhaltend, die Codebasis zu modifizieren.
- [Widerstand gegen Veränderung](widerstand-gegen-veraenderung.md)
<br/>  Teams mit negativen Erfahrungen aus der Vergangenheit widersetzen sich aktiv Refactoring oder Verbesserungen aufgrund wahrgenommenen Risikos basierend auf früheren Fehlschlägen.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Vergangene Vorfälle, die durch Änderungen verursacht wurden, führen dazu, dass Entwickler Refactoring vermeiden, selbst wenn sie anerkennen, dass es notwendig ist.
- [Systemstagnation](systemstagnation.md)
<br/>  Wenn Entwickler aufgrund vergangener Fehlschläge zu vorsichtig sind, Änderungen vorzunehmen, entwickelt sich das System nicht weiter und stagniert.
- [Vermeidungsverhalten](vermeidungsverhalten.md)
<br/>  Entwickler verschieben oder vermeiden die Arbeit an komplexen Bereichen der Codebasis, wo vergangene Änderungen Probleme verursacht haben.
- [Große Schätzungen für kleine Änderungen](grosse-schaetzungen-fuer-kleine-aenderungen.md)
<br/>  Teams liefern überhöhte Schätzungen für Änderungen in Bereichen, wo vergangene Modifikationen Probleme verursacht haben, was exzessive Vorsicht widerspiegelt.

## Causes ▼

- [Schuldzuweisungskultur](schuldzuweisungskultur.md)
<br/>  Wenn Fehler bestraft statt als Lernmöglichkeiten behandelt werden, verinnerlichen Entwickler negative Erfahrungen und werden risikoscheu.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Eine fragile Codebasis, bei der Änderungen häufig unerwartete Brüche verursachen, schafft wiederholte negative Erfahrungen für Entwickler.
- [Geschichte fehlgeschlagener Änderungen](geschichte-fehlgeschlagener-aenderungen.md)
<br/>  Eine Historie fehlgeschlagener Deployments und problematischer Änderungen schafft direkt die negativen Erfahrungen, die Teams vorsichtig machen.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne automatisierte Tests als Sicherheitsnetz sind Änderungen riskant und verursachen mit höherer Wahrscheinlichkeit die Produktionsvorfälle, die negative Erfahrungen schaffen.

## Detection Methods ○
- **Entwicklerbefragungen:** Befragung von Entwicklern zu ihrem Vertrauensniveau bei Änderungen an verschiedenen Teilen des Systems.
- **Änderungshäufigkeitsanalyse:** Überwachung, wie oft verschiedene Module modifiziert werden; konsequent gemiedene Bereiche können auf Angst hinweisen.
- **Schätzungsmuster:** Suche nach Mustern, bei denen ähnliche Änderungen wild unterschiedliche Schätzungen haben, abhängig vom betroffenen Codebereich.
- **Code-Review-Kommentare:** Beobachtung übermäßiger Vorsicht oder langwieriger Diskussionen über potenzielle Risiken während Code-Reviews.

## Examples
Ein Entwickler nimmt eine Änderung an der Codebasis vor, die einen erheblichen Produktionsausfall verursacht. Der Entwickler wird für den Ausfall verantwortlich gemacht und zögert zukünftig, Änderungen an der Codebasis vorzunehmen. Dies ist ein häufiges Problem in Unternehmen mit einer Schuldzuweisungskultur. Es ist wichtig, eine Kultur zu schaffen, in der es sicher ist, zu scheitern. Dies wird Entwickler ermutigen, Risiken einzugehen und zu innovieren.
