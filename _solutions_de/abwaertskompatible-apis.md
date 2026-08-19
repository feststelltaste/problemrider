---
title: Abwärtskompatible APIs
description: Weiterentwicklung von API-Verträgen, ohne bestehende Konsumenten zu
  brechen.
category:
- Architecture
problems:
- breaking-changes
- legacy-api-versioning-nightmare
- api-versioning-conflicts
- poor-interfaces-between-applications
- integration-difficulties
- fear-of-breaking-changes
layout: solution
lang: de
en_slug: backward-compatible-apis
related_solutions:
- slug: backward-compatibility
  similarity: 0.9
- slug: forward-compatibility
  similarity: 0.8
- slug: backward-compatible-data-formats
  similarity: 0.8
- slug: api-deprecation-policy
  similarity: 0.8
- slug: api-first-development
  similarity: 0.75
- slug: api-versioning-strategy
  similarity: 0.75
---

## Description

Abwärtskompatible APIs sind Schnittstellenverträge, die sich nur durch Hinzufügung weiterentwickeln — neue optionale Felder, neue Endpunkte, neue Antwortattribute —, während bestehende Felder, Endpunkte und Statuscodes ihre ursprüngliche Bedeutung und ihr Verhalten unbegrenzt beibehalten, sodass Clients, die gegen eine ältere Version des Vertrags geschrieben wurden, unverändert gegen eine neuere weiterfunktionieren. Der Mechanismus verlässt sich darauf, dass beide Seiten ihren Teil einhalten: Der Server darf nie umnutzen oder entfernen, was bereits existiert, und Konsumenten müssen als tolerante Leser handeln, die Felder ignorieren, die sie nicht erkennen, statt bei ihnen zu versagen. Diese Disziplin ist besonders relevant für Legacy-Systeme, weil ihre APIs häufig über viele Jahre Konsumenten angehäuft haben — interne Services, Partnerintegrationen, Batch-Jobs —, von denen viele schlecht dokumentiert oder dem aktuellen Team völlig unbekannt sind, was einen koordinierten brechenden Rollout über alle hinweg praktisch unmöglich macht. Vertragstests, die die Erwartungen alter Konsumenten kodieren, fungieren als Leitplanke und fangen versehentliche Breaking Changes ab, bevor sie Produktion erreichen, statt nachdem eine Partnerintegration still versagt. Der Tradeoff ist, dass die API über die Zeit veraltete Felder und duale Code-Pfade anhäuft, da nichts jemals sauber entfernt wird ohne einen separaten, bewussten Deprecation-Zyklus — ein Preis, den Legacy-Teams akzeptieren, weil er kleiner ist als der Preis, Integrationen zu brechen, die sie nicht einmal vollständig aufzählen können.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Fügen Sie neue Felder und Endpunkte hinzu, statt bestehende zu ändern oder zu entfernen
- Machen Sie neue Anfragefelder optional mit sinnvollen Standardwerten, sodass bestehende Clients sie nicht senden müssen
- Nutzen Sie tolerante Leser: Konsumenten sollten unbekannte Felder ignorieren, statt bei ihnen zu versagen
- Wenden Sie Vertragstests an, die validieren, dass alte Konsumentenerwartungen nach Änderungen weiterhin gelten
- Vermeiden Sie es, die semantische Bedeutung bestehender Felder oder Statuscodes zu ändern
- Wenn ein Feld seinen Typ oder seine Bedeutung ändern muss, führen Sie ein neues Feld ein und deprekieren Sie das alte

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht kontinuierliche API-Evolution ohne koordinierte Konsumenten-Releases
- Verringert Integrationsfehler und Produktionsvorfälle, die durch Breaking Changes verursacht werden
- Baut Konsumentenvertrauen auf und vereinfacht Partner-Onboarding

**Kosten und Risiken:**
- APIs häufen veraltete Felder und Endpunkte an, was die kognitive Last für neue Entwickler erhöht
- Tolerante-Leser-Muster können echte Bugs im Datenaustausch verbergen
- Die Aufrechterhaltung abwärtskompatiblen Verhaltens in der Geschäftslogik fügt Implementierungskomplexität hinzu
- Erfordert schließlich Bereinigung durch einen formalen Deprecation-Zyklus

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Versicherungsplattform musste ihre Schadensmeldungs-API ändern, um ein neues Dokumentformat zu unterstützen. Statt das bestehende Dokumentfeld zu ändern, fügte das Team ein neues optionales Feld für das strukturierte Format hinzu, während das ursprüngliche Feld funktionsfähig blieb. Bestehende Konsumenten reichten Schäden unverändert weiter ein, während neue Konsumenten sich für das reichhaltigere Format entscheiden konnten. Von Konsumenten gemeldete Fehler sanken während des Übergangs auf null, verglichen mit drei größeren Vorfällen während einer vorherigen brechenden API-Änderung.
