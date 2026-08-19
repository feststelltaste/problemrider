---
title: Interaktive Tutorials
description: Einbettung praktischer, schrittweiser Übungen für Lernaufgaben jeder
  Komplexität.
category:
- Communication
quality_tactics_url: https://qualitytactics.de/en/usability/interactive-tutorials/
problems:
- user-confusion
- user-frustration
- inadequate-onboarding
- poor-user-experience-ux-design
- difficult-developer-onboarding
- increased-customer-support-load
- knowledge-gaps
layout: solution
lang: de
en_slug: interactive-tutorials
related_solutions:
- slug: video-tutorials
  similarity: 0.85
- slug: integrated-onboarding
  similarity: 0.75
- slug: knowledge-base
  similarity: 0.7
- slug: contextual-help
  similarity: 0.7
- slug: personal-support
  similarity: 0.65
- slug: user-communities
  similarity: 0.65
---

## Description

Ein interaktives Tutorial führt einen Nutzer durch eine echte Aufgabe innerhalb der laufenden Anwendung, hebt jedes relevante Element hervor und wartet auf die echte Aktion, bevor es fortschreitet, was weit effektiver lehrt als statische Dokumentation für die komplexen, mehrbildschirmigen Workflows, die Legacy-Systeme besonders neigen anzusammeln. Einen Sandbox- oder Übungsmodus bereitzustellen zählt hier mehr als in einer typischen modernen Anwendung, da Fehler, die beim Erlernen eines Legacy-Workflows auf echten Daten gemacht werden, echte Konsequenzen haben können, die ein neueres, nachsichtigeres System nicht tragen würde. Der Zielkonflikt ist, dass diese Tutorials teuer zu bauen und brüchig zu pflegen sind — jede Änderung an der Oberfläche des tutorialisierten Workflows riskiert, das Schritt-für-Schritt-Skript des Tutorials zu brechen, sodass es sich lohnt, sie für die wirkungsvollsten, fehleranfälligsten Workflows zu reservieren, statt eine umfassende Abdeckung zu versuchen.

## How to Apply ◆

> Interaktive Tutorials gehen über statische Dokumentation hinaus, indem sie Nutzer durch echte Aufgaben innerhalb der laufenden Anwendung führen und praktisches Lernen bieten, das Vertrauen und Kompetenz aufbaut.

- Identifizieren Sie die häufigsten und komplexesten Aufgaben, mit denen neue Nutzer kämpfen. Fokussieren Sie interaktive Tutorials auf diese wirkungsvollen Workflows, statt zu versuchen, jedes Feature abzudecken.
- Bauen Sie schrittweise geführte Erlebnisse, die bei jedem Schritt das relevante UI-Element hervorheben, erklären, was zu tun ist und warum, und darauf warten, dass der Nutzer die Aktion abschließt, bevor fortgeschritten wird.
- Bieten Sie einen Sandbox- oder Übungsmodus, in dem Nutzer Tutorials abschließen können, ohne Produktionsdaten zu beeinflussen. Dies ist besonders wichtig in Legacy-Systemen, in denen Fehler echte Konsequenzen haben können.
- Erlauben Sie Nutzern, Tutorials an jedem Punkt zu verlassen und fortzusetzen. Erzwungene Tutorials, die nicht übersprungen oder pausiert werden können, frustrieren Nutzer, die unterbrochen werden oder bereits Teile des Workflows kennen.
- Beziehen Sie kontextbezogene Erklärungen bei jedem Schritt ein, die über „hier klicken" hinausgehen, um den Zweck und die Geschäftslogik hinter jeder Aktion zu erklären und tieferes Verständnis aufzubauen.
- Machen Sie Tutorials über eine zentrale Bibliothek zugänglich, damit Nutzer sie finden und erneut abspielen können, wenn sie ihr Wissen über einen bestimmten Workflow auffrischen müssen.

## Tradeoffs ⇄

> Interaktive Tutorials bieten die effektivste Form der Nutzerschulung, sind aber am teuersten zu erstellen und zu pflegen.

**Vorteile:**

- Bietet praktisches Lernen, das für komplexe Legacy-System-Workflows deutlich effektiver ist als das Lesen von Dokumentation oder das Ansehen von Videos.
- Verringert die Abhängigkeit von erfahrenen Kollegen für Training, was die Zeit erfahrener Nutzer freisetzt und konsistente Trainingsqualität sicherstellt.
- Baut Nutzervertrauen auf, indem ihnen erlaubt wird, Aufgaben in einer geführten Umgebung zu üben, bevor sie sie an echten Daten ausführen.
- Verringert das Volumen an Support-Anfragen zu „Wie mache ich"-Fragen, indem Selbstbedienungs-Lernressourcen bereitgestellt werden.

**Kosten und Risiken:**

- Interaktive Tutorials sind teuer zu bauen, weil sie Integration mit der laufenden UI und sorgfältiges Schritt-für-Schritt-Skripting für jeden Workflow erfordern.
- Jede Änderung an der UI eines tutorialisierten Workflows kann das Tutorial brechen, was Pflegeaufwand proportional zur Rate der Oberflächenänderungen erfordert.
- Tutorials, die zu lang oder zu detailliert sind, verlieren die Aufmerksamkeit der Nutzer. Jedes Tutorial sollte in fünf bis zehn Minuten abschließbar sein.
- Der Bau von Tutorials für ein Legacy-System mit inkonsistenter UI kann unterschiedliche Tutorial-Mechanismen für verschiedene Teile der Anwendung erfordern.

## How It Could Be

> Komplexe Legacy-System-Workflows, die spezifische Abfolgen von Aktionen über mehrere Bildschirme erfordern, sind erstklassige Kandidaten für interaktive Tutorials.

Ein Legacy-Beschaffungssystem hat einen komplexen Bestellfreigabe-Workflow, der das Erstellen einer Anforderung, das Anhängen von Budgetcodes, die Weiterleitung zur Genehmigung und die Umwandlung in eine Bestellung umfasst. Neue Beschaffungsmitarbeiter machen typischerweise Fehler, die manuelle Korrektur durch Administratoren erfordern. Das Team baut ein interaktives Tutorial, das Nutzer durch die Erstellung einer Beispielbestellung in einer Trainingsumgebung führt, jeden zu klickenden Button und jedes auszufüllende Feld hervorhebt und die Genehmigungs-Routing-Logik bei jedem Entscheidungspunkt erklärt. Neue Mitarbeiter, die das Tutorial abschließen, bevor sie echte Bestellungen bearbeiten, machen im ersten Monat deutlich weniger Fehler, und die von Administratoren für manuelle Korrekturen aufgewendete Zeit sinkt entsprechend. Das Tutorial wird auch als Auffrischung für bestehende Mitarbeiter genutzt, wenn der Workflow aktualisiert wird.
