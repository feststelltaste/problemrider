---
title: On-Site Customer
description: Direkte Einbindung von Kunden in die Entwicklung.
category:
- Requirements
- Communication
problems:
- stakeholder-developer-communication-gap
- requirements-ambiguity
- inadequate-requirements-gathering
- misaligned-deliverables
- customer-dissatisfaction
- feedback-isolation
- no-continuous-feedback-loop
- implementation-rework
layout: solution
lang: de
en_slug: on-site-customer
related_solutions:
- slug: user-stories
  similarity: 0.75
- slug: evolutionary-requirements-development
  similarity: 0.75
- slug: prototypes
  similarity: 0.75
- slug: personas
  similarity: 0.75
- slug: requirements-analysis
  similarity: 0.75
- slug: prototyping
  similarity: 0.75
---

## Description

Ein On-Site Customer ist ein echter Nutzer oder Fachbereichsvertreter, der direkt im Entwicklungsteam eingebettet ist und Fragen zum tatsächlichen Systemverhalten in Minuten statt über langsame, indirekte Kanäle beantworten kann. Diese Praxis ist besonders wertvoll bei der Legacy-Modernisierung, wo die Spezifikation dafür, wie sich das System verhalten soll, häufig das undokumentierte, im Erfahrungswissen der täglichen Nutzer verankerte Wissen ist statt irgendetwas Schriftliches — was bedeutet, dass der On-Site Customer oft die einzige verlässliche Wahrheitsquelle dafür wird, warum ein Arbeitsablauf so funktioniert, wie er funktioniert. Indem Features kontinuierlich gegen reale Nutzungsmuster validiert werden, statt durch späte Abnahmetests, erkennt das Team Missverständnisse, bevor sie sich zu erheblichem Nacharbeitsaufwand summieren, und erfasst die Workarounds und inoffiziellen Prozesse, die sonst vollständig verloren gingen, sobald das Legacy-System abgeschaltet wird.

## How to Apply ◆

> In Legacy-Modernisierungsprojekten verhindert ein im Entwicklungsteam eingebetteter Kundenvertreter den häufigen Fehlerfall, technisch exzellente Ersatzsysteme zu bauen, die tatsächliche Nutzerbedürfnisse verfehlen.

- Identifizieren Sie einen Kunden- oder Nutzervertreter mit tiefem Wissen über die täglichen Arbeitsabläufe des Legacy-Systems und sichern Sie sich dessen Zusage für mindestens mehrere Stunden pro Woche direkter Verfügbarkeit für das Team.
- Platzieren Sie den Kundenvertreter physisch oder virtuell beim Entwicklungsteam, damit Fragen zum Legacy-Verhalten in Minuten statt über tagelangen E-Mail-Austausch beantwortet werden können.
- Lassen Sie den On-Site Customer an Sprint-Planung und Story-Verfeinerung teilnehmen, um Anforderungen zu klären, die aus undokumentiertem Legacy-Verhalten stammen — er weiß oft, warum ein Prozess auf eine bestimmte Weise funktioniert, wenn keine Dokumentation existiert.
- Nutzen Sie den On-Site Customer, um fertiggestellte Features gegen reale Nutzungsmuster zu validieren und Missverständnisse zu erkennen, bevor sie sich zu größerem Nacharbeitsaufwand summieren.
- Ermutigen Sie den Kunden, dem Team tatsächliche Legacy-System-Arbeitsabläufe vorzuführen, einschließlich Workarounds und inoffizieller Prozesse, die in keinem Spezifikationsdokument erscheinen werden.
- Rotieren Sie On-Site-Customer-Vertreter periodisch, um unterschiedliche Perspektiven zu erfassen und Einzelpersonen-Bias bei der Anforderungsinterpretation zu vermeiden.

## Tradeoffs ⇄

> Ein im Team eingebetteter Kunde reduziert Anforderungsmehrdeutigkeit dramatisch, erfordert aber organisatorisches Engagement und sorgfältiges Management der Kundenzeit.

**Vorteile:**

- Beseitigt die Verzögerung zwischen dem Auftreten einer Anforderungsfrage und dem Erhalt einer autoritativen Antwort, was besonders wertvoll ist bei der Modernisierung von Systemen mit undokumentierten Geschäftsregeln.
- Reduziert Implementierungs-Nacharbeit durch frühes Erkennen von Missverständnissen mittels kontinuierlicher Validierung statt später Abnahmetests.
- Baut gemeinsames Verständnis zwischen technischen und fachlichen Stakeholdern auf und reduziert die Kommunikationslücke, die Modernisierungsprojekte häufig entgleisen lässt.
- Erfasst implizites Wissen über die Legacy-System-Nutzung, das sonst verloren ginge, wenn das alte System abgeschaltet wird.

**Kosten und Risiken:**

- Einen Kundenvertreter zu finden, der sowohl tiefes Fachwissen als auch Verfügbarkeit besitzt, um dem Entwicklungsteam erhebliche Zeit zu widmen, kann schwierig sein.
- Ein einzelner On-Site Customer repräsentiert möglicherweise nur eine Perspektive, was zu Lösungen führt, die für seinen Arbeitsablauf funktionieren, aber nicht für andere Nutzergruppen.
- Der Kundenvertreter kann zum Engpass werden, wenn sich das Team für jede Entscheidung auf ihn verlässt, statt eigenes Fachverständnis aufzubauen.
- Organisationspolitik kann verhindern, dass die richtige Person für die Rolle zugewiesen wird, was zu einem Vertreter führt, dem Autorität oder Wissen fehlt.

## How It Could Be

> Die folgenden Szenarien veranschaulichen die Wirkung der Einbindung eines On-Site Customer bei der Legacy-Modernisierung.

Eine Kommunalverwaltung ersetzte ein 25 Jahre altes Genehmigungssystem. Erste Versuche auf Basis schriftlicher Anforderungsdokumente führten zu einem System, das technisch die Spezifikationen erfüllte, aber von Genehmigungssachbearbeitern abgelehnt wurde, weil es sie durch einen starren linearen Arbeitsablauf zwang, statt durch das flexible Jonglieren mit mehreren Anträgen, das sie tatsächlich täglich praktizierten. Nachdem eine erfahrene Genehmigungssachbearbeiterin drei Tage pro Woche in das Entwicklungsteam eingebettet wurde, entdeckte das Team Dutzende undokumentierter Abkürzungen und Workarounds, die essenziell waren, um tägliche Bearbeitungsziele zu erreichen. Die kontinuierliche Einbindung der Sachbearbeiterin reduzierte den Nacharbeitsaufwand um geschätzte 40 % und führte zu einem System, das die Genehmigungsmitarbeiter tatsächlich der Legacy-Anwendung vorzogen.
