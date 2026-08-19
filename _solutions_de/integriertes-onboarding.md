---
title: Integriertes Onboarding
description: Orchestrierung eines ganzheitlichen Erstnutzungserlebnisses mit
  schrittweiser Offenlegung und kontextbezogener Anleitung.
category:
- Communication
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/integrated-onboarding/
problems:
- inadequate-onboarding
- user-confusion
- user-frustration
- difficult-developer-onboarding
- poor-user-experience-ux-design
- increased-customer-support-load
- negative-user-feedback
- new-hire-frustration
- inconsistent-onboarding-experience
- rapid-team-growth
layout: solution
lang: de
en_slug: integrated-onboarding
related_solutions:
- slug: structured-onboarding-program
  similarity: 0.85
- slug: interactive-tutorials
  similarity: 0.75
- slug: empty-states-and-first-use-guidance
  similarity: 0.75
- slug: intuitive-navigation
  similarity: 0.7
- slug: video-tutorials
  similarity: 0.7
- slug: knowledge-base
  similarity: 0.7
---

## Description

Integriertes Onboarding führt einen neuen Nutzer direkt innerhalb der laufenden Anwendung durch seine ersten Schlüsselaktionen — Tooltips, eine geführte Tour, aufgabenbasierte Abschnitte —, statt ihn ein Legacy-Systems veraltete Konventionen rein durch Versuch, Irrtum und die Geduld eines erfahrenen Kollegen erlernen zu lassen. Legacy-Oberflächen sind überproportional schwer auf diese Weise zu erlernen, gerade weil ihre Konventionen aktuellen Designnormen vorausgehen und ein Großteil dessen, wie man sie tatsächlich nutzt, Erfahrungswissen ist, das nirgendwo aufgeschrieben wurde, wo ein neuer Nutzer es finden könnte. Das Onboarding wiederholbar, an das bereits Gelernte anpassungsfähig und rollenspezifisch statt einer generischen Tour durch alles zu gestalten, wandelt eine Abhängigkeit von der Zeit erfahrener Kollegen in ein konsistentes Selbstbedienungserlebnis um — vorausgesetzt, der Inhalt bleibt aktuell, da eine Onboarding-Tour, die auf inzwischen verschobene UI-Elemente zeigt, schlimmer ist als gar keine Tour.

## How to Apply ◆

> Legacy-Systeme sind für neue Nutzer notorisch schwer zu erlernen, weil die Oberflächenkonventionen veraltet sind und Erfahrungswissen erforderlich ist, um sie effektiv zu bedienen. Integriertes Onboarding glättet die Lernkurve.

- Implementieren Sie ein Erstnutzungserlebnis, das bei der ersten Anmeldung des Nutzers aktiviert wird, die wichtigsten Oberflächenelemente hervorhebt und den grundlegenden Arbeitsablauf durch Tooltips, Popover oder eine geführte Tour erklärt.
- Teilen Sie das Onboarding in aufgabenbasierte Abschnitte, statt zu versuchen, das gesamte System auf einmal zu lehren. Führen Sie Nutzer durch ihre erste Schlüsselaktion, wie das Erstellen ihres ersten Datensatzes oder das Abschließen ihrer ersten Transaktion.
- Erlauben Sie Nutzern, die Onboarding-Tour jederzeit über eine Hilfemenüoption erneut abzuspielen. Nutzer, die die anfängliche Tour abbrechen, möchten später vielleicht Anleitung, wenn sie auf unbekannte Bereiche des Systems stoßen.
- Verfolgen Sie den Onboarding-Abschluss und passen Sie das Erlebnis basierend darauf an, was der Nutzer bereits gelernt hat. Erklären Sie keine Konzepte erneut, deren Verständnis der Nutzer bereits demonstriert hat.
- Bieten Sie rollenbasierte Onboarding-Pfade, die sich auf die für die Verantwortlichkeiten jedes Nutzers relevanten Features konzentrieren, statt auf eine generische Tour durch das gesamte System.
- Kombinieren Sie In-App-Onboarding mit Kurzanleitungs-Dokumentation, auf die Nutzer außerhalb der Anwendung für detailliertere Erklärungen zurückgreifen können.

## Tradeoffs ⇄

> Integriertes Onboarding verringert die Zeit bis zur Produktivität neuer Nutzer, erfordert aber Investition in Inhaltserstellung und -pflege.

**Vorteile:**

- Verringert die Zeit, die neue Nutzer brauchen, um produktiv zu werden, dramatisch, was besonders wertvoll für Legacy-Systeme mit steilen Lernkurven ist.
- Verringert die Support-Last während des Onboardings, indem Selbstbedienungsanleitung direkt innerhalb der Anwendung bereitgestellt wird.
- Verringert die Frustration neuer Mitarbeiter, die durch das Fallenlassen in ein unbekanntes Legacy-System ohne Anleitung verursacht wird.
- Schafft ein konsistentes Onboarding-Erlebnis, das nicht von der Verfügbarkeit oder Lehrfähigkeit erfahrener Kollegen abhängt.

**Kosten und Risiken:**

- Onboarding-Inhalt muss aktualisiert werden, wann immer sich die Oberfläche ändert, sonst zeigt er auf Elemente, die nicht mehr existieren oder sich anders verhalten.
- Aufdringliches Onboarding, das erfahrene Nutzer unterbricht oder nicht leicht abgebrochen werden kann, erzeugt Frustration statt sie zu verringern.
- Der Bau interaktiver Touren erfordert Frontend-Entwicklungsaufwand, der mit anderen Prioritäten konkurriert, besonders in Legacy-Systemen mit begrenzten Entwicklungsbudgets.
- Onboarding, das zu viel auf einmal abdeckt, überwältigt neue Nutzer statt ihnen zu helfen, was sorgfältige Abgrenzung dessen erfordert, was einbezogen wird.

## How It Could Be

> Neue Nutzer von Legacy-Systemen beschreiben ihre erste Erfahrung oft so, als würden sie ohne Training ins Cockpit eines Flugzeugs fallengelassen.

Ein Legacy-Projektportfolio-Managementsystem wird organisationsweit genutzt, mit mehrmals jährlich neuen Projektmanagern. Zuvor brauchte jeder neue Nutzer zwei Tage Eins-zu-eins-Training mit einem erfahrenen Kollegen, der sie durch die nicht offensichtliche Navigation und Terminologie des Systems führte. Das Team implementiert eine interaktive Onboarding-Tour, die neue Nutzer durch das Erstellen ihres ersten Projekts, das Hinzufügen von Teammitgliedern und das Einrichten ihres ersten Meilensteins führt. Die Tour nutzt hervorgehobene Tooltips, die auf jedes relevante Oberflächenelement zeigen und in einfacher Sprache erklären, was es tut. Nach der Einführung des Onboarding-Features sinkt die formale Trainingsanforderung von zwei Tagen auf eine halbtägige Sitzung, die fortgeschrittene Themen abdeckt, und neue Projektmanager berichten, sich innerhalb ihrer ersten Woche sicher im Umgang mit den grundlegenden Features zu fühlen.
