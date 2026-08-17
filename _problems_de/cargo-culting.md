---
title: Cargo-Culting
description: Unkritische Übernahme technischer Lösungen, ohne deren zugrunde liegende
  Prinzipien und Kontext zu verstehen
category:
- Architecture
- Process
- Team
related_problems:
- slug: increased-technical-shortcuts
  similarity: 0.65
- slug: workaround-culture
  similarity: 0.65
- slug: copy-paste-programming
  similarity: 0.65
- slug: premature-technology-introduction
  similarity: 0.6
- slug: perfectionist-review-culture
  similarity: 0.6
- slug: cv-driven-development
  similarity: 0.6
solutions:
- architecture-reviews
- boring-technologies
- technical-skills-development
- architecture-governance
- pattern-language
- code-reading-sessions
- internal-technical-coaching
- technology-radar
- pilot-projects
layout: problem
lang: de
en_slug: cargo-culting
---

## Description

Cargo-Culting stellt ein weitverbreitetes Anti-Pattern in der Softwareentwicklung dar, bei dem Teams blindlings Praktiken, Technologien oder architektonische Muster übernehmen, ohne sie kritisch zu bewerten. Dieses Phänomen entspringt einem oberflächlichen Verständnis, das Nachahmung über Verständnis stellt, was zu Lösungen führt, die anspruchsvoll erscheinen, aber grundlegend nicht zum einzigartigen Kontext und den Anforderungen der Organisation passen. Der Begriff stammt aus pazifischen Inselkulturen, die nach dem Zweiten Weltkrieg westliche Praktiken nachahmten, und dient als kraftvolle Metapher für unkritische technologische Nachahmung.

## Indicators ⟡
- Teammitglieder verweisen häufig auf "Best Practices", ohne die Begründung dahinter zu erklären
- Übernahme neuer Technologien oder Muster unmittelbar nach deren Popularitätsgewinn ohne Bewertung
- Kopieren von Code-Lösungen von Stack Overflow oder Tutorials ohne Anpassung
- Umsetzung von Design-Mustern oder architektonischen Stilen, weil "erfolgreiche Unternehmen das so machen"
- Befolgen von Prozess-Zeremonien oder Methodiken, ohne deren Zweck zu verstehen
- Das Team kann nicht erklären, warum bestimmte Praktiken oder Werkzeuge gewählt wurden, außer "es wird empfohlen"
- Widerstand gegen das Hinterfragen oder Anpassen übernommener Praktiken, selbst wenn sie nicht zum Kontext passen

## Symptoms ▲

- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Schlecht passende übernommene Lösungen erfordern Workarounds, um sie an den tatsächlichen Problemkontext anzupassen.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Übernommene Technologien und Muster, die das Team nicht versteht, werden teuer in Wartung und Fehlerbehebung.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Unangemessen komplexe Architekturen, die ohne Verständnis übernommen wurden, verlangsamen die Feature-Lieferung.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Code, der Muster verwendet, die das Team nicht wirklich versteht, wird schwer zu lesen, zu ändern und zu debuggen.
- [Probleme mit algorithmischer Komplexität](probleme-mit-algorithmischer-komplexitaet.md)
<br/>  Entwickler, die Code-Muster kopieren, ohne deren Performance-Eigenschaften zu verstehen, können ineffiziente Algorithmen einführen.
- [Ausrichtungs- und Padding-Probleme](ausrichtungs-und-padding-probleme.md)
<br/>  Das Kopieren von Struct-Definitionen ohne Verständnis ihrer Speicherauswirkungen führt zu suboptimaler Feldreihenfolge und suboptimalem Layout.

## Causes ▼

- [Unvollständiges Wissen](unvollstaendiges-wissen.md)
<br/>  Unzureichendes Verständnis zugrunde liegender Prinzipien führt dazu, dass Teams Lösungen kopieren, statt angemessene zu entwerfen.
- [CV-getriebene Entwicklung](cv-getriebene-entwicklung.md)
<br/>  Entwickler übernehmen angesagte Technologien, um ihren Lebenslauf aufzuwerten, statt Lösungen zu wählen, die zum Problem passen.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter Termindruck übernehmen Teams bestehende Lösungen vollständig, statt Zeit zu investieren, um sie zu verstehen und anzupassen.

## Detection Methods ○
- **Warum-Interviews:** Durchführung von Interviews, in denen Teammitglieder gebeten werden, die Begründung hinter technischen Entscheidungen zu erklären
- **Entscheidungsdokumentation:** Überprüfung von Entscheidungsprotokollen zur Verifikation der Begründung über externe Referenzen hinaus
- **Code-Komplexitätsanalyse:** Identifikation übermäßig komplexer Muster, die nicht zur Komplexität des Problems passen
- **Performance-Monitoring:** Nachverfolgung von Performance-Metriken nach der Implementierung neuer Technologien
- **Musterkonsistenzprüfungen:** Verifikation der konsistenten Implementierung von Mustern im gesamten System
- **Quellenrückverfolgung:** Identifikation von Code, der direkt aus Tutorials kopiert wurde, ohne sinnvolle Anpassung
- **Änderungsschwierigkeit:** Beobachtung von Bereichen, in denen sich das Team schwertut, bestehende Lösungen zu ändern
- **Trendanalyse:** Vergleich der Technologieübernahme mit breiteren Branchentrends
- **Troubleshooting-Bewertung:** Bewertung der Fähigkeit des Teams, Probleme in übernommenen Lösungen eigenständig zu lösen

## Examples

Ein Entwicklungsteam liest über den erfolgreichen Einsatz von Microservices-Architektur bei großen Tech-Unternehmen und beschließt, seine monolithische Anwendung in Dutzende kleiner Services aufzuteilen. Es hat jedoch nicht die operative Infrastruktur, die Teamgröße oder die Organisationsstruktur, um Microservices effektiv zu unterstützen. Das Ergebnis ist ein verteilter Monolith mit der gesamten Komplexität von Microservices, aber keinem der Vorteile. Die Netzwerklatenz steigt, das Debugging wird viel schwieriger, und die Deployment-Komplexität vervielfacht sich. Auf die Frage, warum es diesen Ansatz gewählt hat, kann das Team nur auf Blogbeiträge großer Tech-Unternehmen verweisen, ohne artikulieren zu können, wie sich der eigene Kontext unterscheidet oder welche spezifischen Probleme mit der architektonischen Änderung gelöst werden sollten.
