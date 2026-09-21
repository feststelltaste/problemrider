---
title: Gemischte Coding-Stile
description: Eine Situation, in der unterschiedliche Teile der Codebasis unterschiedliche
  Formatierung, Namenskonventionen und Designmuster verwenden.
category:
- Code
related_problems:
- slug: inconsistent-codebase
  similarity: 0.75
- slug: inconsistent-coding-standards
  similarity: 0.75
- slug: inconsistent-naming-conventions
  similarity: 0.7
- slug: undefined-code-style-guidelines
  similarity: 0.7
- slug: code-duplication
  similarity: 0.6
- slug: difficult-code-reuse
  similarity: 0.6
solutions:
- static-analysis-and-linting
- code-conventions
- style-guide
- code-reviews
- code-review-guidelines
- code-quality-gates
- ci-cd-pipeline
- clean-code
- quality-ratchet
- automated-code-migration
- large-scale-refactoring
layout: problem
lang: de
en_slug: mixed-coding-styles
---

## Description
Gemischte Coding-Stile sind eine Situation, in der unterschiedliche Teile der Codebasis unterschiedliche Formatierung, Namenskonventionen und Designmuster verwenden. Dies ist ein häufiges Problem in lang laufenden Projekten, besonders solchen, an denen im Laufe der Jahre viele verschiedene Personen gearbeitet haben. Gemischte Coding-Stile können zu einer Reihe von Problemen führen, einschließlich verringerter Lesbarkeit, erhöhter kognitiver Last und einer allgemeinen Verlangsamung der Entwicklungsgeschwindigkeit.

## Indicators ⟡
- Die Codebasis ist schwer zu lesen und zu verstehen.
- Es gibt mehrere Wege, dasselbe zu tun.
- Die Codebasis ist eine Mischung aus verschiedenen Stilen und Konventionen.
- Es gibt keinen Style Guide für das Projekt, oder er existiert, wird aber nicht durchgesetzt.

## Symptoms ▲

- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Inkonsistente Formatierung, Benennung und Muster zwingen Entwickler zu mentalem Kontextwechsel, was Code schwerer lesbar und verständlich macht.
- [Stildiskussionen in Code-Reviews](stildiskussionen-in-code-reviews.md)
<br/>  Ohne konsistenten Stil verkommen Code-Reviews zu Debatten über Formatierungs- und Benennungspräferenzen statt über Logik und Design.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Entwickler verbringen zusätzliche Zeit damit, inkonsistente Code-Muster zu entziffern, was das Tempo der Feature-Lieferung verlangsamt.
- [Frustration neuer Mitarbeiter](frustration-neuer-mitarbeiter.md)
<br/>  Neue Entwickler, die dem Projekt beitreten, sind von inkonsistenten Konventionen verwirrt und kämpfen damit, zu wissen, welchem Stil sie folgen sollen.
- [Inkonsistente Codebasis](inkonsistente-codebasis.md)
<br/>  Gemischte Coding-Stile tragen direkt zu einer insgesamt inkonsistenten Codebasis bei, der es an Kohärenz über Module hinweg fehlt.

## Causes ▼

- [Undefinierte Code-Stil-Richtlinien](undefinierte-code-stil-richtlinien.md)
<br/>  Ohne definierte und durchgesetzte Stilrichtlinien wendet jeder Entwickler seine eigenen bevorzugten Konventionen an.
- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Häufige Entwicklerfluktuation bringt neue Personen mit unterschiedlichen Coding-Gewohnheiten hervor, jede hinterlässt ihren stilistischen Abdruck in der Codebasis.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Das Fehlen vereinbarter oder durchgesetzter Coding-Standards führt direkt dazu, dass unterschiedliche Teile der Codebasis unterschiedliche Stile nutzen.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Ohne gründliche Code-Reviews, die Konsistenz durchsetzen, häufen sich Stilabweichungen über die Zeit ungebremst an.

## Detection Methods ○
- **Manuelle Code-Inspektion:** Die Inkonsistenz ist oft offensichtlich, wenn man einfach durch die Codebasis blättert.
- **Ausführen eines Linters oder Formatters:** Ausführen eines Werkzeugs wie ESLint, Prettier, RuboCop oder Black auf der Codebasis und Beobachtung der großen Anzahl gemeldeter Verstöße.
- **Team-Befragungen:** Befragung von Entwicklern, ob sie die Codebasis leicht lesbar und verständlich finden.
- **Analyse von Code-Review-Kommentaren:** Suche nach hoher Häufigkeit von Kommentaren zu Stil und Formatierung.

## Examples
Eine große Unternehmensanwendung wurde über ein Jahrzehnt von mehreren Teams entwickelt. Ein Modul nutzt camelCase für Variablen, ein anderes nutzt snake_case, und ein drittes mischt beides. Einrückung variiert zwischen Tabs und Leerzeichen, und Klammerstile sind inkonsistent. Dies macht es für jeden einzelnen Entwickler sehr schwierig, effizient über Module hinweg zu arbeiten. Ein neuer Entwickler tritt bei und reicht einen Pull Request ein, der mehrfach aufgrund von Stilverstößen abgelehnt wird, die nie explizit kommuniziert wurden, was zu Frustration und Verzögerungen führt.
