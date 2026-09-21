---
title: Breaking Changes
description: API-Aktualisierungen brechen bestehende Client-Integrationen, verursachen
  Kompatibilitätsprobleme und erzwingen kostspielige Notfall-Fixes.
category:
- Architecture
- Code
- Testing
related_problems:
- slug: api-versioning-conflicts
  similarity: 0.7
- slug: legacy-api-versioning-nightmare
  similarity: 0.65
- slug: rapid-system-changes
  similarity: 0.65
- slug: change-management-chaos
  similarity: 0.6
- slug: increasing-brittleness
  similarity: 0.6
- slug: regression-bugs
  similarity: 0.6
solutions:
- anti-corruption-layer
- dependency-management-strategy
- adapter
- api-deprecation-policy
- api-versioning-strategy
- backward-compatibility
- backward-compatible-apis
- backward-compatible-data-formats
- backward-compatible-schema-migrations
- compatibility-as-error
- compatibility-governance
- compatibility-measurement
- compatibility-requirements
- compatibility-standards
- compatibility-testing
- consumer-driven-contracts
- content-negotiation
- continuous-integration
- cross-platform-serialization
- cross-version-testing
- dependency-pinning
- forward-compatibility
- interoperability-tests
- schema-registry
- semantic-versioning
- tolerant-reader
- version-control
- versioning-scheme
- third-party-dependency-check
layout: problem
lang: de
en_slug: breaking-changes
---

## Description

Breaking Changes entstehen, wenn Änderungen an APIs, Schnittstellen oder Systemverhalten dazu führen, dass bestehende Client-Integrationen fehlschlagen oder sich falsch verhalten. Diese Änderungen verletzen Erwartungen an Abwärtskompatibilität und zwingen Clients dazu, ihren Code zu aktualisieren, oft unerwartet und kurzfristig. Breaking Changes können Beziehungen zu Integrationspartnern erheblich schädigen, Produktionsausfälle verursachen und Notfall-Support-Situationen schaffen.

## Indicators ⟡

- Client-Anwendungen funktionieren nach API-Updates nicht mehr
- Integrationspartner melden plötzliche Ausfälle in ihren Systemen
- Support-Tickets steigen unmittelbar nach API-Releases sprunghaft an
- Client-Entwickler äußern Frustration über unerwartete Änderungen
- Notfall-Rollbacks sind nötig, um die Client-Funktionalität wiederherzustellen

## Symptoms ▲

- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Breaking Changes an APIs verursachen Ausfälle abhängiger Services in einer Kettenreaktion, da jeder versucht, die geänderte Schnittstelle zu nutzen.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Zuvor funktionierende Client-Integrationen beginnen nach API-Änderungen Fehler zu zeigen, da ihre Annahmen gebrochen werden.
- [Unzufriedenheit der Stakeholder](unzufriedenheit-der-stakeholder.md)
<br/>  Integrationspartner und Kunden verlieren Vertrauen, wenn ihre Systeme aufgrund unerwarteter API-Änderungen brechen.
- [Budgetüberschreitungen](budgetueberschreitungen.md)
<br/>  Notfall-Fixes und ungeplante Client-Migrationsarbeit durch Breaking Changes treiben die Kosten über den Plan hinaus.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Wiederholte Vorfälle von Breaking Changes erzeugen organisatorische Angst vor jeder künftigen API-Änderung.
- [ABI-Kompatibilitätsprobleme](abi-kompatibilitaetsprobleme.md)
<br/>  Wenn Breaking Changes die exportierten Funktionssignaturen oder Datenlayouts einer kompilierten Bibliothek ohne ordentliche Versionierung ändern, verursachen sie direkt ABI-Kompatibilitätsprobleme; Änderungen an nicht-binären Schnittstellen (z. B. REST-APIs) verursachen stattdessen API-Level-Inkompatibilitäten statt ABI-Problemen.

## Causes ▼

- [API-Versionierungskonflikte](api-versionierungskonflikte.md)
<br/>  Schlechte API-Versionierungspraktiken machen es unmöglich, APIs weiterzuentwickeln, ohne bestehende Konsumenten zu brechen.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Fehlende Integrationstests gegen bestehende Client-Nutzungsmuster lassen Breaking Changes vor der Veröffentlichung unentdeckt bleiben.
- [Chaos im Change-Management](chaos-im-change-management.md)
<br/>  Änderungen, die ohne ordentliche Koordination oder Auswirkungsbewertung deployt werden, brechen Client-Integrationen unerwartet.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Ohne dokumentierte API-Verträge wissen Entwickler nicht, auf welches Verhalten sich Clients verlassen, und können es unbeabsichtigt brechen.

## Detection Methods ○

- **Integrationstest-Monitoring:** Automatisierte Tests, die die API-Kompatibilität mit bestehenden Client-Mustern verifizieren
- **Client-Nutzungsanalytik:** Beobachtung, wie verschiedene API-Endpunkte und Parameter tatsächlich genutzt werden
- **Versionskompatibilitätstests:** Testen neuer API-Versionen gegen bestehenden Client-Code und Integrationsmuster
- **Client-Feedback-Kanäle:** Einrichtung von Kommunikationskanälen, über die Clients Kompatibilitätsprobleme melden können
- **Änderungsauswirkungsbewertung:** Systematische Bewertung, wie sich vorgeschlagene Änderungen auf bestehende Integrationen auswirken
- **Breaking-Change-Warnungen:** Automatisierte Erkennung von Änderungen, die bestehenden Client-Code brechen könnten

## Examples

Eine E-Commerce-API ändert die Datenstruktur von Produktinformations-Antworten und verschiebt das Preisfeld von einer einfachen Zahl zu einem komplexen Objekt mit Währungs- und Steuerinformationen. Hunderte Client-Anwendungen, die das Preisfeld direkt parsen, brechen sofort, was Fehler im Warenkorb und der Bestellverarbeitung über mehrere Einzelhandels-Websites hinweg verursacht. Der API-Anbieter muss sowohl das alte als auch das neue Antwortformat aufrechterhalten, während Clients hektisch ihren Code aktualisieren. Ein weiteres Beispiel betrifft eine Zahlungsabwicklungs-API, die Authentifizierungsanforderungen ohne ausreichende Vorankündigung ändert, was dazu führt, dass alle Client-Transaktionen während der Haupteinkaufszeiten fehlschlagen, was zu Millionenverlusten bei Verkäufen und Notfall-Support-Anrufen führt.
