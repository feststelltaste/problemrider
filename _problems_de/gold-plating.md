---
title: Gold Plating
description: Entwickler fügen einem Projekt unnötige Features oder Komplexität hinzu,
  weil sie glauben, dies werde die Stakeholder beeindrucken, selbst wenn es nicht
  angefordert wurde.
category:
- Process
related_problems:
- slug: incomplete-projects
  similarity: 0.7
- slug: feature-creep
  similarity: 0.65
- slug: large-feature-scope
  similarity: 0.65
- slug: feature-creep-without-refactoring
  similarity: 0.6
- slug: slow-feature-development
  similarity: 0.6
- slug: budget-overruns
  similarity: 0.6
solutions:
- architecture-reviews
- boring-technologies
- technical-skills-development
- explicit-prioritization-framework
- feature-usage-measurement
- definition-of-done
- definition-of-ready
- story-mapping
- regular-stakeholder-demonstrations
- outcome-based-goal-setting
- fit-to-standard-principle
layout: problem
lang: de
en_slug: gold-plating
---

## Description
Gold Plating ist die Praxis, einem Projekt unnötige Features oder Komplexität hinzuzufügen. Dies geschieht oft durch Entwickler, die glauben, dass es die Stakeholder beeindrucken wird. Gold Plating kann jedoch zu einer Reihe von Problemen führen, darunter Scope Creep, Feature-Creep und ein aufgeblähtes, unfokussiertes Produkt. Gold Plating ist ein verbreitetes Problem in der Softwareentwicklung und lässt sich nur schwer vermeiden.

## Indicators ⟡
- Das Team fügt Features hinzu, die nicht angefordert wurden.
- Das Team verbringt viel Zeit mit Features, die nicht essenziell sind.
- Das Produkt wird im Laufe der Zeit immer komplexer.
- Das Team ist nicht auf das Minimum Viable Product (MVP) fokussiert.

## Symptoms ▲

- [Scope Creep](scope-creep.md)
<br/>  Das Hinzufügen nicht angeforderter Features weitet den Projektumfang über die ursprünglichen Grenzen hinaus aus, ohne ordentliche Änderungskontrolle.
- [Verpasste Termine](verpasste-termine.md)
<br/>  Zeit, die für unnötige Features aufgewendet wird, verzögert die Lieferung der eigentlich angeforderten Kernanforderungen.
- [Budgetüberschreitungen](budgetueberschreitungen.md)
<br/>  Ressourcen, die für nicht angeforderte Features aufgewendet werden, verbrauchen Budget, das für essenzielle Arbeit vorgesehen war.
- [Feature-Aufblähung](feature-aufblaehung.md)
<br/>  Unnötige Features häufen sich an, was das Produkt übermäßig komplex macht und sein Kernwertversprechen verwässert.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Jedes unnötige Feature fügt Wartungslast und Testanforderungen hinzu, die die laufenden Entwicklungskosten erhöhen.

## Causes ▼

- [Übermäßige Gefälligkeit gegenüber Stakeholdern](uebermaessige-gefaelligkeit-gegenueber-stakeholdern.md)
<br/>  Der Wunsch, Stakeholder zu beeindrucken, motiviert Entwickler dazu, zusätzliche Features hinzuzufügen, von denen sie glauben, sie würden geschätzt werden.
- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Ohne klare Anforderungen füllen Entwickler wahrgenommene Lücken mit ihren eigenen Annahmen darüber, was wertvoll wäre.
- [Kein formaler Änderungskontrollprozess](kein-formaler-aenderungskontrollprozess.md)
<br/>  Ohne formale Umfangskontrolle können Entwickler nicht angeforderte Features ohne Genehmigung oder Abwägung hinzufügen.

## Detection Methods ○
- **Feature-Anfrage-Backlog:** Analyse des Feature-Anfrage-Backlogs, um Features zu identifizieren, die nicht angefordert wurden.
- **Produkt-Roadmap:** Überprüfung der Produkt-Roadmap, ob sie fokussiert und realistisch ist.
- **Nutzerfeedback:** Anhören von Nutzerfeedback, um zu erkennen, ob das System als komplex und verwirrend empfunden wird.
- **Code-Komplexitätsmetriken:** Nutzung statischer Analysewerkzeuge, um die Komplexität der Codebasis zu messen.

## Examples
Ein Team baut eine neue Website. Das Team entscheidet sich, eine Reihe von Features hinzuzufügen, die von den Stakeholdern nicht angefordert wurden. Das Team glaubt, dass diese Features die Stakeholder beeindrucken werden. Die Features sind jedoch nicht essenziell und fügen der Website viel Komplexität hinzu. Die Website wird verspätet und über Budget geliefert. Die Stakeholder sind nicht beeindruckt.
