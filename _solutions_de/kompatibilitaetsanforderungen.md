---
title: Kompatibilitätsanforderungen
description: Explizite und verbindliche Formulierung impliziter Kompatibilitätsannahmen.
category:
- Requirements
- Process
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- breaking-changes
- integration-difficulties
- fear-of-breaking-changes
- implicit-knowledge
- legal-disputes
layout: solution
lang: de
en_slug: compatibility-requirements
related_solutions:
- slug: documentation-of-compatibility-requirements
  similarity: 0.85
- slug: compatibility-standards
  similarity: 0.8
- slug: compatibility-as-error
  similarity: 0.8
- slug: compatibility-governance
  similarity: 0.8
- slug: compatibility-measurement
  similarity: 0.8
- slug: compatibility-testing
  similarity: 0.8
---

## Description

Kompatibilitätsanforderungen sind die explizite, schriftliche Spezifikation, mit welchen externen Systemen, Protokollversionen und Datenformaten ein System kompatibel bleiben muss, was eine sonst unformulierte Annahme, nur in den Köpfen weniger langjähriger Ingenieure getragen, in eine dokumentierte, verifizierbare Zusage verwandelt. Einmal aufgeschrieben, können diese Anforderungen direkt an Nutzergeschichten und Akzeptanzkriterien angehängt werden statt nur als unausgesprochener Hintergrund funktionaler Anforderungen zu leben, und Testfälle können direkt daraus abgeleitet werden, was Kompatibilität zu etwas macht, das das Team aktiv verifiziert, statt etwas, von dem sie lediglich hoffen, es bewahrt zu haben. Diese Dokumentationslücke ist üblich in Legacy-Systemen, die sich über lange Zeiträume mit vielen Partnersystemen integrieren, wo die spezifische Protokollversion oder das Datenformat, auf das sich jeder Partner verlässt, zur Zeit der ursprünglichen Integration informell verstanden, aber nirgendwo dauerhaft erfasst wurde, was aktuelle Betreuer ohne verlässliche Möglichkeit zurücklässt zu wissen, was eine routinemäßig aussehende Änderung brechen könnte. Wenn diese Lücke existiert, kann ein scheinbar gewöhnliches Upgrade still eine Annahme verletzen, an die sich niemand zu prüfen erinnerte, was mehrere Partnerintegrationen gleichzeitig bricht und die fehlende Anforderung erst während der folgenden Vorfallüberprüfung offenbart. Kompatibilitätsanforderungen als ständigen Teil von Architektur-Reviews zu überprüfen und Integrationspartner direkt in ihre Definition und Validierung einzubeziehen hält die dokumentierten Anforderungen mit dem abgestimmt, was Partner tatsächlich brauchen, statt dem, was das Team annimmt, dass sie brauchen. Die offensichtlichen Kosten sind der laufende Aufwand, diese Dokumentation über viele Partner hinweg zu sammeln und zu pflegen und sie aktuell zu halten, während sich Bedürfnisse weiterentwickeln, und es gibt eine organisatorische Tendenz, sich dagegen zu wehren, implizite Annahmen explizit zu machen, genau weil dies klare Verantwortlichkeit schafft, wo zuvor keine existierte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Dokumentieren Sie, mit welchen Systemen, Versionen, Protokollen und Datenformaten Ihr System kompatibel bleiben muss
- Beziehen Sie Kompatibilitätsanforderungen in Nutzergeschichten und Akzeptanzkriterien ein, nicht nur in funktionale Anforderungen
- Leiten Sie Testfälle direkt aus Kompatibilitätsanforderungen ab, sodass sie verifizierbar sind
- Überprüfen Sie Kompatibilitätsanforderungen während Architektur-Reviews und vor größeren Änderungen
- Pflegen Sie ein lebendiges Dokument von Kompatibilitätszusagen, zugänglich für alle Teams
- Beziehen Sie Integrationspartner in die Definition und Validierung von Kompatibilitätsanforderungen ein

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verhindert Kompatibilitätsprobleme, die durch unformulierte Annahmen verursacht werden
- Gibt Entwicklern klare Anleitung darüber, was sie während Änderungen bewahren müssen
- Schafft eine vertragliche Grundlage für Kompatibilitätstests und -validierung

**Kosten und Risiken:**
- Das Sammeln und Pflegen von Kompatibilitätsanforderungen kostet Aufwand und teamübergreifende Koordination
- Übermäßig starre Anforderungen können notwendige architektonische Evolution einschränken
- Anforderungen könnten veralten, wenn sie nicht regelmäßig überprüft werden
- Stakeholder könnten sich dagegen wehren, implizite Annahmen explizit zu machen, weil dies Verantwortlichkeit schafft

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Transportunternehmen integrierte sich mit 15 Partnersystemen, hatte aber nie dokumentiert, welche Protokollversionen und Datenformate jeder Partner benötigte. Als ein routinemäßiges Upgrade drei Partnerintegrationen brach, offenbarte die Vorfallüberprüfung, dass keine Kompatibilitätsanforderungen existierten. Das Team verbrachte zwei Wochen damit, Anforderungen für alle Partnerintegrationen zu dokumentieren, fügte sie den Architecture Decision Records hinzu und erstellte automatisierte Kompatibilitätstests. In den folgenden 12 Monaten traten keine ungeplanten Partnerintegrationsfehler auf.
