---
title: Unzureichende Design-Fähigkeiten
description: Dem Entwicklungsteam fehlen die nötigen Fähigkeiten und Erfahrung, um
  gut strukturierte, wartbare Software zu entwerfen und zu bauen.
category:
- Code
- Team
related_problems:
- slug: inexperienced-developers
  similarity: 0.7
- slug: insufficient-testing
  similarity: 0.65
- slug: insufficient-code-review
  similarity: 0.65
- slug: inappropriate-skillset
  similarity: 0.65
- slug: incomplete-knowledge
  similarity: 0.65
- slug: skill-development-gaps
  similarity: 0.65
solutions:
- architecture-reviews
- boring-technologies
- solid-principles
- technical-skills-development
- pattern-language
- refactoring-katas
- lightweight-design-review
- internal-technical-coaching
- communities-of-practice
- code-reading-sessions
- domain-driven-design
layout: problem
lang: de
en_slug: insufficient-design-skills
---

## Description
Unzureichende Design-Fähigkeiten sind ein wesentlicher Beitragender zur Entstehung von Legacy-Code. Wenn einem Entwicklungsteam die nötigen Fähigkeiten und Erfahrung fehlen, um gut strukturierte, wartbare Software zu entwerfen und zu bauen, schafft es wahrscheinlich ein System, das schwer zu verstehen, zu modifizieren und zu testen ist. Dies kann zu einer Reihe von Problemen führen, einschließlich einer hohen Fehlerrate, langsamer Entwicklungsgeschwindigkeit und erheblicher Frustration für das Entwicklungsteam. Unzureichende Design-Fähigkeiten sind ein verbreitetes Problem in der Softwarebranche und können schwer anzugehen sein.

## Indicators ⟡
- Die Codebasis ist ein "großer Klumpen Schlamm".
- Das Team kämpft ständig mit technischen Schulden.
- Das Team kann neue Features nicht termingerecht liefern.
- Das Team ist nicht stolz auf den Code, den es schreibt.

## Symptoms ▲

- [Spaghetticode](spaghetticode.md)
<br/>  Fehlende Design-Fähigkeiten führen zu schlecht strukturiertem Code mit verworrenem Kontrollfluss und unklarer Organisation.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Entwickler ohne Design-Fähigkeiten schaffen eng gekoppelte Module mit schlechter Trennung der Belange.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Schlechte Designentscheidungen häufen technische Schulden an, die zunehmend teurer werden, sie zu beheben.
- [Schwierige Code-Wiederverwendung](schwierige-code-wiederverwendung.md)
<br/>  Schlecht gestaltete Komponenten sind schwer wiederzuverwenden, weil ihnen klare Schnittstellen und ordentliche Abstraktionen fehlen.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Schlecht gestaltete Systeme werden zunehmend schwerer zu modifizieren, was die Feature-Entwicklung verlangsamt.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Ohne ordentliches Design brechen Änderungen in einem Bereich häufig andere Teile des Systems.

## Causes ▼

- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Junior-Entwickler, die noch keine Softwaredesign-Prinzipien gelernt haben, fehlen naturgemäß Design-Fähigkeiten.
- [Unzureichende Mentoring-Struktur](unzureichende-mentoring-struktur.md)
<br/>  Ohne Mentoring erhalten Entwickler nicht die Anleitung, die nötig ist, um Design-Fähigkeiten über die Zeit zu entwickeln.
- [Lücken in der Kompetenzentwicklung](luecken-in-der-kompetenzentwicklung.md)
<br/>  Organisationen, die nicht in Schulung investieren, lassen Entwickler ohne Gelegenheiten, Design-Kompetenzen aufzubauen.
- [Termindruck](termindruck.md)
<br/>  Ständiger Termindruck verhindert, dass Entwickler ordentliche Designpraktiken lernen und anwenden.

## Detection Methods ○
- **Code-Reviews:** Code-Reviews sind ein guter Weg, um Designprobleme zu identifizieren.
- **Code-Komplexitätsmetriken:** Nutzung statischer Analysewerkzeuge zur Messung der Komplexität der Codebasis.
- **Entwicklerbefragungen:** Befragung von Entwicklern zu ihrem Vertrauen in ihre Design-Fähigkeiten.
- **Architekturbewertungen:** Durchführung einer Bewertung der Systemarchitektur zur Identifikation von Designfehlern.

## Examples
Ein Unternehmen stellt ein Team von Junior-Entwicklern ein, um eine neue Webanwendung zu bauen. Die Entwickler haben keine Erfahrung mit Softwaredesign und haben keinen Mentor, der sie anleitet. Infolgedessen schaffen sie ein System, das schlecht gestaltet und schwer zu warten ist. Das Unternehmen kann neue Features nicht termingerecht liefern und kämpft ständig mit Fehlern. Das Unternehmen muss schließlich ein Team erfahrener Entwickler einstellen, um das gesamte System neu zu schreiben.
