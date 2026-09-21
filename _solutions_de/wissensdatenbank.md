---
title: Wissensdatenbank
description: Aufbau einer durchsuchbaren Wissensdatenbank mit Artikeln, Anleitungen
  und Problemlösungen.
category:
- Communication
quality_tactics_url: https://qualitytactics.de/en/usability/knowledge-base/
problems:
- poor-documentation
- increased-customer-support-load
- knowledge-gaps
- user-confusion
- knowledge-silos
- implicit-knowledge
- difficult-developer-onboarding
- information-fragmentation
- duplicated-effort
- duplicated-research-effort
- extended-research-time
- high-turnover
- inadequate-mentoring-structure
- inconsistent-onboarding-experience
- knowledge-sharing-breakdown
- mentor-burnout
- team-churn-impact
- unclear-documentation-ownership
- unproductive-meetings
- communication-breakdown
- duplicated-work
- incomplete-knowledge
- inconsistent-knowledge-acquisition
- knowledge-dependency
- poor-communication
- information-decay
- language-barriers
- unclear-sharing-expectations
layout: solution
lang: de
en_slug: knowledge-base
related_solutions:
- slug: knowledge-sharing-practices
  similarity: 0.85
- slug: user-communities
  similarity: 0.8
- slug: frequently-asked-questions-faq
  similarity: 0.75
- slug: personal-support
  similarity: 0.75
- slug: runbooks
  similarity: 0.75
- slug: contextual-help
  similarity: 0.75
---

## Description

Eine Wissensdatenbank ist eine durchsuchbare Sammlung von Artikeln, konsistent strukturiert und aus den häufigsten Support-Anfragen aufgebaut, die das Erfahrungswissen erfasst, das Legacy-Systeme in den Köpfen der wenigen Personen ansammeln, die sie am längsten genutzt haben. Weil dieses Wissen typischerweise nirgendwo sonst existiert, ist jeder Weggang eines erfahrenen Nutzers oder Entwicklers effektiv eine kleine Wissenskrise, und der gesamte Wert der Datenbank kommt daher, dieses Wissen aufzuschreiben, bevor es zur Tür hinausgeht, statt danach. Ihr Nutzen hängt vollständig davon ab, aktuell zu bleiben: Eine ungepflegte Wissensdatenbank mit veralteten Artikeln führt Nutzer aktiv in die Irre und untergräbt genau das Vertrauen, das sie aufbauen soll, weshalb ein schlanker Beitrags- und Review-Workflow so wichtig ist wie der ursprüngliche Inhalt selbst.

## How to Apply ◆

> Legacy-Systeme sammeln riesige Mengen an Erfahrungswissen an, das nur in den Köpfen erfahrener Nutzer und Entwickler existiert. Eine durchsuchbare Wissensdatenbank erfasst und teilt dieses Wissen systematisch.

- Beginnen Sie damit, die Lösungen für die häufigsten Support-Anfragen zu dokumentieren. Analysieren Sie den Support-Ticket-Verlauf, um die Top-Probleme zu identifizieren, und schreiben Sie klare Artikel, die jedes davon adressieren.
- Strukturieren Sie Artikel mit einer konsistenten Vorlage, die Problembeschreibung, Schritt-für-Schritt-Lösung, Screenshots oder Diagramme und verwandte Artikel enthält. Konsistenz macht die Wissensdatenbank leichter durchsuchbar und navigierbar.
- Implementieren Sie Volltextsuche mit Relevanzranking, damit Nutzer Artikel mit ihren eigenen Worten finden können, statt der exakten Terminologie, die in den Artikeltiteln genutzt wird.
- Markieren und kategorisieren Sie Artikel nach Thema, Nutzerrolle und Systemmodul, damit Nutzer relevante Inhalte durchsuchen können, selbst wenn sie nicht sicher sind, wonach sie suchen sollen.
- Etablieren Sie einen Beitrags- und Review-Workflow, der es Support-Personal und erfahrenen Nutzern erleichtert, neue Artikel einzureichen und bestehende zu aktualisieren, ohne Engpässe.
- Verfolgen Sie Artikelansichten, Suchanfragen ohne Ergebnisse und Nutzerbewertungen, um Abdeckungslücken und verbesserungsbedürftige Artikel zu identifizieren.
- Verlinken Sie Wissensdatenbankartikel von relevanten Stellen innerhalb der Anwendung, wie Fehlermeldungen, Hilfe-Tooltips und Onboarding-Abläufen.

## Tradeoffs ⇄

> Eine Wissensdatenbank demokratisiert den Zugang zu Systemwissen, erfordert aber anhaltenden Aufwand, um aktuell und umfassend zu bleiben.

**Vorteile:**

- Verringert das Support-Ticket-Volumen, indem Nutzern erlaubt wird, Antworten unabhängig zu finden, bevor sie den Helpdesk kontaktieren.
- Erfasst institutionelles Wissen, das sonst verloren ginge, wenn erfahrene Teammitglieder gehen, was Wissenssilos und Risiken impliziten Wissens abmildert.
- Bietet eine konsistente, maßgebliche Informationsquelle, die die Variation in Qualität und Genauigkeit informeller mündlicher Erklärungen beseitigt.
- Beschleunigt das Onboarding, indem neuen Nutzern eine Selbstbedienungsressource gegeben wird, um das System in ihrem eigenen Tempo zu erlernen.

**Kosten und Risiken:**

- Eine ungepflegte Wissensdatenbank mit veralteten Artikeln führt Nutzer aktiv in die Irre und untergräbt das Vertrauen in die Ressource.
- Der Aufbau einer umfassenden Wissensdatenbank erfordert erheblichen Vorabaufwand, um bestehende Prozesse und Lösungen zu dokumentieren.
- Die Wissensdatenbank kann ein falsches Gefühl von Dokumentationsvollständigkeit erzeugen, was Teams dazu bringt, ihre Aktualisierung zu vernachlässigen, während sich das System weiterentwickelt.
- Ohne Analytik und Feedback-Mechanismen kann das Team nicht erkennen, welche Artikel hilfreich sind und welche fehlen oder unzureichend sind.

## How It Could Be

> Wenn Wissen über ein Legacy-System nur als Erfahrungswissen existiert, erzeugt jeder Weggang eines erfahrenen Teammitglieds eine Wissenskrise.

Ein Legacy-Fertigungsausführungssystem wurde über ein Jahrzehnt von einem kleinen Team gepflegt. Als der leitende Entwickler in Rente geht, entdeckt das verbleibende Team, dass Dutzende kritische Betriebsverfahren nur in den persönlichen Notizen und im Gedächtnis des pensionierten Entwicklers dokumentiert waren. Das Team startet eine Wissensdatenbank-Initiative, beginnend mit den dringendsten Lücken: Systemstartverfahren, häufige Fehlerbehebungsschritte und Konfigurationsleitfäden. Sie interviewen verbleibende erfahrene Nutzer und Entwickler, um deren Wissen zu erfassen, bevor es verloren geht. Innerhalb von sechs Monaten enthält die Wissensdatenbank über zweihundert Artikel, die die häufigsten Betriebs- und Fehlerbehebungsszenarien abdecken. Neue Teammitglieder können nun häufige Probleme unabhängig lösen, indem sie die Wissensdatenbank durchsuchen, und die durchschnittliche Lösungszeit für bekannte Probleme sinkt, weil Support-Personal nicht mehr die eine Person finden und konsultieren muss, die zufällig die Antwort kennt.
