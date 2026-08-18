---
title: Inkonsistente Codebasis
description: Dem Code des Projekts fehlt ein einheitlicher Stil, Coding-Standards
  und Designmuster, was ihn schwer lesbar, wartbar und für das Onboarding neuer Entwickler
  erschwert.
category:
- Code
- Process
related_problems:
- slug: inconsistent-coding-standards
  similarity: 0.85
- slug: undefined-code-style-guidelines
  similarity: 0.8
- slug: inconsistent-naming-conventions
  similarity: 0.8
- slug: mixed-coding-styles
  similarity: 0.75
- slug: code-duplication
  similarity: 0.7
- slug: difficult-to-understand-code
  similarity: 0.7
solutions:
- static-analysis-and-linting
- architecture-conformity-analysis
- architecture-governance
- architecture-review-board
- bubble-context
- code-conventions
- code-generation
- consistent-user-interface
- design-tokens
- fluent-interfaces
- pattern-language
- fitness-functions
- style-guide
layout: problem
lang: de
en_slug: inconsistent-codebase
---

## Description
Einer inkonsistenten Codebasis fehlt kohärentes und einheitliches Design, Stil und Standards. Dies äußert sich auf mehrere Weisen: unterschiedliche Namenskonventionen und Coding-Stile, variierende Formatierungs- und Strukturmuster, gemischte Einrückungsstile, inkonsistente Klammerstile und das Vorhandensein mehrerer konkurrierender Implementierungen derselben Funktionalität. Wenn jeder Entwickler seinen eigenen Konventionen folgt, ist das Ergebnis eine chaotische und unvorhersehbare Codebasis, die schwer zu verstehen, zu warten und zu erweitern wird. Eine inkonsistente Codebasis ist eine wesentliche Quelle technischer Schulden und ein Hindernis für wirksame Zusammenarbeit. Das Etablieren und Durchsetzen konsistenter Coding-Standards ist essenziell, um ein wartbares System zu schaffen.

## Indicators ⟡
- Es ist schwierig, sich in der Codebasis zurechtzufinden.
- Man muss oft andere Entwickler um Hilfe bitten, um den Code zu verstehen.
- Es gibt mehrere Wege, dasselbe zu tun.
- Die Codebasis ist eine Mischung aus unterschiedlichen Stilen und Konventionen.
- Es gibt keinen Styleguide für das Projekt, oder er existiert, wird aber nicht durchgesetzt.
- Es gibt häufige Auseinandersetzungen über Stil in Code-Reviews.

## Symptoms ▲

- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Gemischte Stile und Muster erschweren es Entwicklern, Code über unterschiedliche Module hinweg zu lesen und zu verstehen.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Entwickler haben Schwierigkeiten, produktiv zu werden, weil es keine konsistenten Muster gibt, die sie über die Codebasis hinweg lernen und anwenden können.
- [Stildiskussionen in Code-Reviews](stildiskussionen-in-code-reviews.md)
<br/>  Ohne vereinbarte Standards verkommen Code-Reviews zu Debatten über stilistische Präferenzen.
- [Erhöhte kognitive Last](erhoehte-kognitive-last.md)
<br/>  Entwickler müssen beim Arbeiten über Module hinweg mental zwischen unterschiedlichen Konventionen und Mustern wechseln.
- [Inkonsistentes Verhalten](inkonsistentes-verhalten.md)
<br/>  Eine inkonsistente Codebasis, in der für dieselben Belange unterschiedliche Muster genutzt werden, führt direkt zu inkonsistentem Systemverhalten.

## Causes ▼

- [Undefinierte Code-Stil-Richtlinien](undefinierte-code-stil-richtlinien.md)
<br/>  Ohne klare Coding-Standards folgt jeder Entwickler seinen eigenen Präferenzen, was zu inkonsistentem Code führt.
- [Auswirkung von Team-Fluktuation](auswirkung-von-team-fluktuation.md)
<br/>  Während Entwickler über die Zeit kommen und gehen, bringt jeder unterschiedliche Coding-Konventionen mit, die sich in der Codebasis anhäufen.
- [Unzureichende Code-Reviews](unzureichende-code-reviews.md)
<br/>  Oberflächliche Reviews versäumen es, konsistente Coding-Standards durchzusetzen, und erlauben es, dass Stilinkonsistenzen bestehen bleiben.
- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Ohne klare Eigenverantwortung für Codequalitätsstandards übernimmt niemand die Verantwortung für die Wahrung von Konsistenz.

## Detection Methods ○

- **Manuelle Code-Inspektion:** Die Inkonsistenz ist oft offensichtlich, wenn man einfach durch die Codebasis browst. Manuelle Inspektion unterschiedlicher Teile der Codebasis zur Identifikation stilistischer Variationen.
- **Ausführung eines Linters oder Formatters:** Ausführung eines Werkzeugs wie ESLint, Prettier, RuboCop oder Black auf der Codebasis und Beobachtung der großen Anzahl gemeldeter Verstöße.
- **Teambefragungen:** Befragung von Entwicklern, ob sie die Codebasis leicht lesbar und verständlich finden, und zu ihrer Erfahrung mit Code-Lesbarkeit und -Konsistenz.
- **Analyse der Code-Review-Kommentare:** Suche nach hoher Häufigkeit von Kommentaren zu Stil und Formatierung. Beobachtung wiederkehrender Kommentare zu Stil während Code-Reviews.

## Examples
Ein Entwickler versucht, einen Fehler in einem Legacy-Modul zu beheben. Er stellt fest, dass das Modul eine völlig andere Namenskonvention für Variablen und Funktionen nutzt als der Rest der Anwendung. Dies erschwert es, den Code zu verstehen und zuversichtlich zu sein, dass sein Fix keine unbeabsichtigten Nebeneffekte hat. In einem anderen Fall hat ein Projekt zwei unterschiedliche Module, die sich beide mit einer Datenbank verbinden müssen. Ein Modul nutzt eine Connection-Pool-Bibliothek, während das andere für jede Abfrage eine neue Verbindung öffnet und schließt. Diese Inkonsistenz macht die Anwendung schwerer zu konfigurieren und zu debuggen.

Eine große Enterprise-Anwendung wurde über ein Jahrzehnt von mehreren Teams entwickelt. Ein Modul nutzt camelCase für Variablen, ein anderes snake_case, und ein drittes mischt beides. Die Einrückung variiert zwischen Tabs und Leerzeichen, und Klammerstile sind inkonsistent. Dies macht es für jeden einzelnen Entwickler sehr schwierig, effizient über Module hinweg zu arbeiten. Ein neuer Entwickler tritt bei und reicht einen Pull Request ein, der mehrfach aufgrund von Stilverstößen abgelehnt wird, die nie explizit kommuniziert wurden, was zu Frustration und Verzögerungen führt. Dies ist ein sehr verbreitetes Problem in lang laufenden Projekten, besonders solchen, an denen über die Jahre viele unterschiedliche Personen gearbeitet haben. Es ist ein klassisches Zeichen technischer Schulden, das Wartbarkeit, Zusammenarbeit und die Gesamtcodequalität erheblich beeinträchtigt.
