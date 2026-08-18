---
title: Unerfahrenheit der Reviewer
description: Reviewern fehlt die Erfahrung, um tiefere Probleme zu identifizieren,
  sodass sie sich auf das konzentrieren, was sie verstehen.
category:
- Code
- Culture
- Process
- Team
related_problems:
- slug: inadequate-code-reviews
  similarity: 0.8
- slug: insufficient-code-review
  similarity: 0.75
- slug: inexperienced-developers
  similarity: 0.75
- slug: inadequate-initial-reviews
  similarity: 0.75
- slug: team-members-not-engaged-in-review-process
  similarity: 0.75
- slug: reviewer-anxiety
  similarity: 0.75
solutions:
- code-review-process-reform
- pair-and-mob-programming
- code-review-guidelines
- checklists
- knowledge-rotation
- structured-onboarding-program
- technical-skills-development
- internal-technical-coaching
layout: problem
lang: de
en_slug: reviewer-inexperience
---

## Description
Unerfahrenheit der Reviewer tritt auf, wenn Teammitglieder, die mit Code-Review beauftragt sind, nicht über die nötigen Fähigkeiten oder das nötige Wissen verfügen, um tiefes, aufschlussreiches Feedback zu geben. Dies führt oft zu Reviews, die entweder übermäßig auf triviale Stilfragen fokussiert sind oder einfach ohne gründliche Analyse der Code-Logik, Architektur oder potenzieller Randfälle abgesegnet werden. Dies kann ein falsches Sicherheitsgefühl schaffen und erlauben, dass kritische Probleme in die Codebasis gelangen.

## Indicators ⟡
- Code-Reviews bestimmter Teammitglieder sind konsequent kurz und ohne substantielle Kommentare.
- Junior-Entwickler werden ohne Anleitung von Senior-Teammitgliedern mit dem Review komplexer Änderungen betraut.
- Es gibt kein formelles Training oder Mentoring-Programm zur Verbesserung der Code-Review-Fähigkeiten.

## Symptoms ▲

- [Reviewer-Angst](reviewer-angst.md)
<br/>  Unerfahrene Reviewer fühlen sich unsicher und ängstlich bezüglich ihrer Fähigkeit, aussagekräftiges Feedback zu geben.
- [Zusammenbruch des Review-Prozesses](zusammenbruch-des-review-prozesses.md)
<br/>  Wenn Reviewern die Erfahrung fehlt, echte Probleme zu identifizieren, werden Reviews oberflächlich und verbessern die Codequalität nicht.
- [Übereilte Genehmigungen](uebereilte-genehmigungen.md)
<br/>  Unerfahrene Reviewer, die echte Probleme nicht identifizieren können, neigen dazu, Änderungen schnell zu genehmigen, statt zuzugeben, dass sie den Code nicht verstehen.
- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Reviews durch unerfahrene Reviewer übersehen kritische Bugs und Design-Mängel, die dann in Produktion gelangen.

## Causes ▼

- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Ein insgesamt unerfahrenes Entwicklungsteam hat naturgemäß unerfahrene Reviewer, denen die Tiefe fehlt, um Codequalität zu bewerten.
- [Unzureichende Mentoring-Struktur](unzureichende-mentoring-struktur.md)
<br/>  Ohne Mentoring-Programme zur Entwicklung von Review-Fähigkeiten bleiben Teammitglieder unerfahren in der Durchführung effektiver Reviews.
- [Wissenslücken](wissensluecken.md)
<br/>  Lücken im Fachwissen oder technischen Verständnis hindern Reviewer daran, tiefere Probleme in unvertrauten Codebereichen zu erkennen.

## Detection Methods ○
- **Analyse der Review-Kommentare:** Suche nach Mustern oberflächlicher oder nicht-substantieller Kommentare von bestimmten Reviewern.
- **Verfolgung von Bug-Ursprüngen:** Rückverfolgung von Produktionsfehlern zu den Code-Änderungen, die sie eingeführt haben, und Prüfung der zugehörigen Code-Reviews.
- **Team-Fähigkeitsbewertung:** Bewertung des gesamten Erfahrungsniveaus des Teams und Identifikation von Wissenslücken.

## Examples
Ein Junior-Entwickler wird gebeten, einen Pull Request zu reviewen, der komplexe Datenbankabfragen betrifft. Aus Mangel an Erfahrung in diesem Bereich konzentriert er sich auf Code-Formatierung und Variablenbenennung und genehmigt den Pull Request. Die ineffizienten Abfragen werden erst später entdeckt, als sie einen Performance-Engpass in Produktion verursachen. Dieses Szenario verdeutlicht, wie Unerfahrenheit die Wirksamkeit von Code-Reviews als Qualitätstor untergraben kann.
