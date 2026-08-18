---
title: Unerfahrene Entwickler
description: Dem Entwicklungsteam fehlt das Wissen und die Erfahrung, um Best Practices
  und wartbare Lösungen zu implementieren.
category:
- Code
- Communication
- Process
related_problems:
- slug: difficult-developer-onboarding
  similarity: 0.75
- slug: reviewer-inexperience
  similarity: 0.75
- slug: insufficient-design-skills
  similarity: 0.7
- slug: inadequate-mentoring-structure
  similarity: 0.7
- slug: knowledge-gaps
  similarity: 0.65
- slug: inappropriate-skillset
  similarity: 0.65
solutions:
- pair-and-mob-programming
- structured-onboarding-program
- refactoring-katas
- security-training
- code-reading-sessions
- internal-technical-coaching
- communities-of-practice
- lightweight-design-review
- code-review-guidelines
- knowledge-rotation
layout: problem
lang: de
en_slug: inexperienced-developers
---

## Description

Unerfahrene Entwickler bezeichnet eine Situation, in der Teammitgliedern das Wissen, die Fähigkeiten oder die Erfahrung fehlen, die nötig sind, um Best Practices der Softwareentwicklung zu implementieren, wartbaren Code zu schreiben oder fundierte architektonische Entscheidungen zu treffen. Dieses Problem ist besonders akut in Legacy-Systemen, in denen komplexe Geschäftslogik und veraltete Technologien sowohl Domänenwissen als auch technische Expertise erfordern. Wenn unerfahrene Entwickler ohne ordentliche Anleitung an komplexen Systemen arbeiten, schaffen sie oft Lösungen, die kurzfristig funktionieren, aber langfristige Wartungsprobleme verursachen.

## Indicators ⟡
- Code-Reviews offenbaren häufig grundlegende Programmierfehler oder Antipatterns
- Neue Teammitglieder unterschätzen durchgängig die Komplexität von Aufgaben
- Lösungen sind übermäßig simplistisch oder übermäßig komplex für das vorliegende Problem
- Grundlegende Softwareentwicklungsprinzipien werden nicht konsistent befolgt
- Das Team verlässt sich stark auf Senior-Entwickler für Anleitung bei Routineaufgaben

## Symptoms ▲

- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Entwicklern ohne Erfahrung führen mit höherer Wahrscheinlichkeit Defekte durch Missverständnis von Code oder Geschäftslogik ein.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Unerfahrene Entwickler produzieren oft Code mit Antipatterns, schlechter Struktur und inkonsistenten Praktiken.
- [Zunehmende technische Abkürzungen](zunehmende-technische-abkuerzungen.md)
<br/>  Unter Lieferdruck greifen Entwickler ohne Erfahrung standardmäßig auf den einfachsten Ansatz zurück, den sie kennen, statt auf einen ordentlich gestalteten, weil sie möglicherweise bessere Alternativen oder die langfristigen Konsequenzen nicht erkennen.
- [Ineffizienter Code](ineffizienter-code.md)
<br/>  Ohne Wissen über Performance-Optimierung schreiben unerfahrene Entwickler rechnerisch teuren Code.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Unerfahrene Entwickler duplizieren häufig Code, weil sie bestehende Implementierungen nicht kennen oder nicht verstehen, wie man bestehende Logik ordentlich abstrahiert und wiederverwendet.
- [Probleme mit algorithmischer Komplexität](probleme-mit-algorithmischer-komplexitaet.md)
<br/>  Entwicklern ohne Grundlagen der Informatik erkennen möglicherweise keine schlechten algorithmischen Entscheidungen oder kennen keine besseren Alternativen.
- [Ausrichtungs- und Padding-Probleme](ausrichtungs-und-padding-probleme.md)
<br/>  Entwickler, die mit Hardware-Speicherausrichtungsanforderungen nicht vertraut sind, schaffen unwissentlich ineffiziente Strukturlayouts.
- [Overhead durch atomare Operationen](overhead-durch-atomare-operationen.md)
<br/>  Entwickler, die mit den Nuancen nebenläufiger Programmierung nicht vertraut sind, nutzen möglicherweise atomare Operationen übermäßig, ohne deren Performance-Kosten zu verstehen.
- [Schwachstellen zur Umgehung der Authentifizierung](schwachstellen-zur-umgehung-der-authentifizierung.md)
<br/>  Entwickler ohne Sicherheitsexpertise implementieren oft benutzerdefinierte Authentifizierungslogik mit subtilen Fehlern, die eine Umgehung erlauben.

## Causes ▼

- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Hohe Fluktuation bedeutet, dass erfahrene Entwickler gehen und durch weniger erfahrene ersetzt werden, was die Gesamtexpertise des Teams verringert.
- [Lücken in der Kompetenzentwicklung](luecken-in-der-kompetenzentwicklung.md)
<br/>  Fehlende Schulungsprogramme und Mentoring-Möglichkeiten hindern Junior-Entwickler daran, ihre Fähigkeiten zu entwickeln.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Wenn Onboarding keine Best Practices und Systemkonventionen lehrt, können besonders Junior-Mitarbeiter weit über ihr Startdatum hinaus unterqualifiziert bleiben, während erfahrene Mitarbeiter trotz langsamen Onboardings typischerweise trotzdem einarbeiten.
- [Unzureichende Mentoring-Struktur](unzureichende-mentoring-struktur.md)
<br/>  Ohne erfahrene Mentoren, die sie anleiten, haben Junior-Entwickler keinen Weg, ordentliche Engineering-Fähigkeiten zu entwickeln.

## Detection Methods ○
- **Codequalitätsmetriken:** Beobachtung von Metriken wie zyklomatischer Komplexität, Code-Abdeckung und Fehlerdichte nach Entwickler
- **Code-Review-Muster:** Nachverfolgung von Häufigkeit und Arten von Problemen, die während Code-Reviews gefunden werden
- **Aufgabenerledigungsanalyse:** Vergleich geschätzter vs. tatsächlicher Zeit für unterschiedliche Entwickler bei ähnlichen Aufgaben
- **Fehlerzuordnung:** Analyse, welche Entwickler die meisten Fehler oder die schwerwiegendsten Probleme einführen
- **Wissensbewertungen:** Regelmäßige technische Bewertungen zur Identifikation von Fähigkeitslücken

## Examples

Ein Finanzdienstleistungsunternehmen stellt mehrere Junior-Entwickler ein, um an einem Legacy-Trading-System zu arbeiten. Das bestehende System nutzt komplexe domänenspezifische Algorithmen für Risikoberechnung, aber die erfahrenen Entwickler sind zu beschäftigt mit neuen Projekten, um ordentliche Anleitung zu geben. Die Junior-Entwickler implementieren neue Features, indem sie bestehende Muster kopieren, ohne die zugrunde liegende Geschäftslogik zu verstehen. Sie schaffen ein neues Risikoberechnungsmodul, das für normale Marktbedingungen korrekte Ergebnisse produziert, aber bei Marktvolatilität katastrophal versagt. Der Fehler wird erst entdeckt, als das System während eines Marktabschwungs erheblich Geld verliert. Die Grundursache war, dass die Junior-Entwickler die mathematischen Modelle hinter den Risikoberechnungen nicht verstanden und eine vereinfachte Version implementierten, die für grundlegende Testfälle funktionierte, aber die ausgefeilte Randfall-Behandlung des ursprünglichen Systems vermissen ließ. Ein weiteres Beispiel betrifft eine Webanwendung, bei der unerfahrene Entwickler Nutzerauthentifizierung implementieren, indem sie Passwörter im Klartext speichern und vorhersehbare Session-Tokens nutzen, weil sie Sicherheits-Best-Practices nicht verstehen, was eine massive Sicherheitslücke schafft.
