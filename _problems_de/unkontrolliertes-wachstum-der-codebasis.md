---
title: Unkontrolliertes Wachstum der Codebasis
description: Eine Situation, in der eine Codebasis in Größe und Komplexität ohne
  jegliche Kontrolle oder Planung wächst.
category:
- Code
related_problems:
- slug: feature-creep-without-refactoring
  similarity: 0.7
- slug: feature-creep
  similarity: 0.7
- slug: unbounded-data-growth
  similarity: 0.65
- slug: brittle-codebase
  similarity: 0.6
- slug: rapid-team-growth
  similarity: 0.6
- slug: large-feature-scope
  similarity: 0.6
solutions:
- architecture-reviews
- clean-code
- loose-coupling
- separation-of-concerns
- solid-principles
- strategic-code-deletion
- tree-shaking
- deprecation-strategy
layout: problem
lang: de
en_slug: uncontrolled-codebase-growth
---

## Description
Unkontrolliertes Wachstum der Codebasis ist eine Situation, in der eine Codebasis in Größe und Komplexität ohne jegliche Kontrolle oder Planung wächst. Dies ist ein häufiges Problem in langlebigen Projekten, wo konstant neue Features hinzugefügt werden, ohne über das Gesamtdesign des Systems nachzudenken. Unkontrolliertes Wachstum der Codebasis kann zu einer Reihe von Problemen führen, einschließlich hoher technischer Schulden, aufgeblähter Klassen und einer allgemeinen Verlangsamung der Entwicklungsgeschwindigkeit.

## Indicators ⟡
- Die Codebasis wird zunehmend größer und komplexer.
- Es dauert immer länger, neue Features hinzuzufügen.
- Die Anzahl der Bugs nimmt zu.
- Entwickler werden zunehmend frustrierter mit der Codebasis.

## Symptoms ▲

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Unkontrolliertes Wachstum fügt Komplexität ohne Designbetrachtung hinzu, was stetig technische Schulden anhäuft.
- [Spaghetticode](spaghetticode.md)
<br/>  Wachstum ohne strukturelle Planung führt zu verworrenem, unstrukturiertem Code, der schwer zu verstehen oder zu modifizieren ist.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Während die Codebasis unkontrolliert wächst, vervielfachen sich gegenseitige Abhängigkeiten, und der Code wird fragil und bruchanfällig.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Eine exzessiv große und komplexe Codebasis verlangsamt die Entwicklung, während Teams mehr Zeit mit Navigation und Verständnis des Codes verbringen.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Eine große, schlecht strukturierte Codebasis macht es erheblich schwieriger, Bugs zu lokalisieren und zu diagnostizieren.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Entwickler stehen vor einer steilen Lernkurve, wenn die Codebasis unkontrolliert groß und komplex geworden ist.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Eine große, komplexe und schlecht strukturierte Codebasis ist erheblich teurer in der Wartung, was eine direkte und vorhersehbare Konsequenz ist.

## Causes ▼

- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Das kontinuierliche Hinzufügen von Features ohne Refactoring ist ein primärer Treiber unkontrollierten Wachstums der Codebasis.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Die Vermeidung von Refactoring bedeutet, dass sich die Codebasis toten Code, redundante Logik und unnötige Komplexität anhäuft.
- [Zeitdruck](zeitdruck.md)
<br/>  Zeitdruck entmutigt Bereinigung und sorgfältiges Design, was die Codebasis ohne strukturelle Verbesserung wachsen lässt.
- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Ohne klare Code-Eigentümerschaft übernimmt niemand Verantwortung dafür, die Codebasis sauber und gut organisiert zu halten.

## Detection Methods ○
- **Code-Metrik-Werkzeuge:** Nutzung von Werkzeugen zur Messung von Codekomplexität, Klassengröße und anderen Metriken.
- **Code-Reviews:** Suche nach Code, der schwer zu verstehen und zu reviewen ist.
- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen zur Identifikation von Code-Smells, wie großen Klassen und langen Methoden.

## Examples
Ein Unternehmen hat eine große, monolithische E-Commerce-Anwendung, die seit über 10 Jahren in Entwicklung ist. Die Codebasis ist auf über eine Million Codezeilen gewachsen, und es wird zunehmend schwieriger, sie zu warten und zu erweitern. Das Entwicklungsteam verbringt immer mehr Zeit mit der Behebung von Bugs und immer weniger Zeit mit dem Hinzufügen neuer Features. Das Unternehmen beginnt, Marktanteile an seine Wettbewerber zu verlieren, die schneller innovieren können.
