---
title: Kontextbezogene Hilfe
description: Bereitstellung von Hilfeinformationen und Erklärungen direkt im aktuellen
  Aufgabenkontext.
category:
- Communication
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/contextual-help/
problems:
- user-confusion
- user-frustration
- poor-user-experience-ux-design
- poor-documentation
- difficult-developer-onboarding
- increased-customer-support-load
- negative-user-feedback
- knowledge-gaps
layout: solution
lang: de
en_slug: contextual-help
related_solutions:
- slug: personal-support
  similarity: 0.75
- slug: knowledge-base
  similarity: 0.75
- slug: understandable-error-messages
  similarity: 0.75
- slug: intuitive-navigation
  similarity: 0.7
- slug: video-tutorials
  similarity: 0.7
- slug: interactive-tutorials
  similarity: 0.7
---

## Description

Kontextbezogene Hilfe bettet Anleitungen genau an der Stelle in der Oberfläche ein, an der ein Nutzer sie braucht — ein Tooltip an einem verwirrenden Feld, eine Inline-Erklärung neben einer kryptischen Fehlermeldung —, statt sich auf ein separates Handbuch zu verlassen, das die meisten Nutzer nie öffnen. Legacy-Systeme haben tendenziell genau die Felder und Arbeitsabläufe, die diese Hilfe am meisten brauchen: Eigenheiten, deren Zweck nur dem ursprünglichen Entwickler klar war und die den Menschen, die das System täglich benutzen, jetzt ohne jede Erklärung präsentiert werden. Hilfeinhalte nah am Element zu platzieren, das sie beschreiben, und sie aktuell zu halten, während sich das System weiterentwickelt, verwandelt eine häufige Ursache für Support-Tickets in etwas, das Nutzer im Moment selbst lösen können.

## How to Apply ◆

> Legacy-Systeme haben oft entweder gar keine Hilfe oder ein separates Hilfehandbuch, das Nutzer nie konsultieren. Kontextbezogene Hilfe bettet Anleitungen direkt dort ein, wo Nutzer sie brauchen, was Verwirrung und Support-Anfragen reduziert.

- Fügen Sie Tooltip-Erklärungen zu Formularfeldern, Schaltflächen und Oberflächenelementen hinzu, die häufig Verwirrung oder Support-Anfragen verursachen. Nutzen Sie Daten aus Support-Tickets, um die wichtigsten Ziele zu identifizieren.
- Implementieren Sie Inline-Hilfetexte für komplexe Felder, die erklären, wofür das Feld ist, welches Format erwartet wird und welche Konsequenzen unterschiedliche Werte haben. Legacy-Systeme haben oft Felder, deren Zweck nur dem ursprünglichen Entwickler klar ist.
- Erstellen Sie kontextsensitive Hilfebereiche, die relevante Anleitung basierend auf dem aktuellen Bildschirm und der aktuellen Aufgabe des Nutzers anzeigen, statt zu verlangen, dass Nutzer ein separates Hilfesystem durchsuchen.
- Fügen Sie Fehlermeldungen erklärenden Text hinzu, der Nutzern nicht nur sagt, was schiefgelaufen ist, sondern auch, was zu tun ist. Legacy-Fehlermeldungen zeigen oft kryptische Codes oder technischen Fachjargon an.
- Nutzen Sie progressive Offenlegung für Hilfeinhalte: Zeigen Sie standardmäßig kurze Hinweise und bieten Sie Zugang zu detaillierten Erklärungen für Nutzer, die mehr Informationen brauchen.
- Halten Sie Hilfeinhalte nah am Element, das sie beschreiben. Nutzer sollten ihren aktuellen Kontext nicht verlassen müssen, um eine Erklärung zu finden.

## Tradeoffs ⇄

> Kontextbezogene Hilfe liefert unmittelbare Antworten am Bedarfspunkt, erfordert aber laufende Pflege, während sich das System weiterentwickelt.

**Vorteile:**

- Reduziert das Support-Ticket-Volumen, indem häufige Fragen direkt in der Oberfläche beantwortet werden, bevor Nutzer sich melden.
- Verringert die Einarbeitungszeit, weil neue Nutzer das System während der Nutzung lernen können, statt ein separates Handbuch zu studieren.
- Behebt Wissenslücken, die durch schlechte oder veraltete Dokumentation entstehen, indem aktuelle, korrekte Anleitung dort platziert wird, wo sie am wichtigsten ist.
- Baut Nutzervertrauen auf, indem an Entscheidungspunkten Rückversicherung geboten wird, was Zögern und Fehler reduziert.

**Kosten und Risiken:**

- Hilfeinhalte müssen zusammen mit der Anwendung gepflegt werden. Veraltete kontextbezogene Hilfe, die Verhalten beschreibt, das sich geändert hat, ist schlimmer als keine Hilfe.
- Übermäßige Tooltips und Inline-Hilfe können die Oberfläche überladen und erfahrene Nutzer stören, die keine Anleitung brauchen, was sorgfältige Abwägung erfordert.
- Effektive Hilfeinhalte zu schreiben erfordert ein detailliertes Verständnis der Nutzeraufgaben, was möglicherweise Zusammenarbeit mit Fachexperten erfordert.
- Die Übersetzung kontextbezogener Hilfe in mehrere Sprachen erhöht den Lokalisierungsaufwand bei international eingesetzten Legacy-Systemen.

## How It Could Be

> Legacy-Systeme leiden oft unter einer Dokumentationslücke, bei der nur die Menschen, die die Oberfläche gebaut haben, sie vor Jahren verstanden haben.

Ein Legacy-Buchhaltungssystem hat einen „Periodenabschluss"-Prozess, der das Setzen von Flags über mehrere Bildschirme in einer bestimmten Reihenfolge beinhaltet. Der Prozess ist in einem vierzigseitigen prozeduralen Handbuch dokumentiert, das Buchhaltungsmitarbeiter ausdrucken und Schritt für Schritt jeden Monat befolgen. Als das Team kontextbezogene Hilfebereiche zu jedem Bildschirm im Periodenabschluss-Workflow hinzufügt, die zeigen, was dieser konkrete Schritt bewirkt und was als Nächstes kommt, können die Buchhaltungsmitarbeiter den Prozess durchführen, ohne das Handbuch zu konsultieren. Neue Buchhalter, die zuvor einen erfahrenen Kollegen brauchten, der sie durch ihre ersten mehreren Monatsabschlüsse führte, können den Prozess jetzt eigenständig mit der eingebetteten Anleitung durchführen. Das Support-Team berichtet, dass periodenabschlussbezogene Fragen, die zuvor jeden Monatsende in die Höhe schossen, fast vollständig eliminiert sind.
