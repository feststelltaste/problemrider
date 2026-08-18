---
title: Schlechtes Vertragsdesign
description: Rechtliche Vereinbarungen und Verträge spiegeln nicht die Projektrealitäten,
  technischen Anforderungen wider oder erlauben nicht die während der Entwicklung
  notwendige Flexibilität.
category:
- Management
- Process
- Security
related_problems:
- slug: legal-disputes
  similarity: 0.65
- slug: process-design-flaws
  similarity: 0.6
- slug: complex-implementation-paths
  similarity: 0.55
- slug: wasted-development-effort
  similarity: 0.55
- slug: scope-change-resistance
  similarity: 0.55
- slug: insufficient-design-skills
  similarity: 0.55
solutions:
- contract-testing
- api-first-development
- compatibility-certification
- vendor-management-practice
- service-level-agreements
- requirements-traceability-matrix
- consumer-driven-contracts
- security-requirements-definition
layout: problem
lang: de
en_slug: poor-contract-design
---

## Description

Schlechtes Vertragsdesign tritt auf, wenn rechtliche Vereinbarungen, die Softwareentwicklungsprojekte regeln, ohne ausreichendes Verständnis technischer Realitäten, Entwicklungsprozesse oder des Bedarfs an Flexibilität während der Implementierung verfasst werden. Diese Verträge enthalten oft unrealistische Liefergegenstände, unflexible Bedingungen, unzureichende Bestimmungen zum Änderungsmanagement oder fehlausgerichtete Anreize, die Probleme während der Projektausführung schaffen.

## Indicators ⟡

- Vertragsbedingungen entsprechen nicht der technischen Machbarkeit oder bewährten Entwicklungspraktiken
- Keine Bestimmungen zur Handhabung von Umfangsänderungen oder Anforderungsentwicklung
- Zahlungspläne stimmen nicht mit Entwicklungsmeilensteinen oder der Fertigstellung von Liefergegenständen überein
- Vertragsstrafen entmutigen notwendige Änderungen oder Qualitätsverbesserungen
- Rechtliche Bedingungen widersprechen technischen oder operativen Anforderungen

## Symptoms ▲

- [Widerstand gegen Scope-Änderungen](widerstand-gegen-scope-aenderungen.md)
<br/>  Starre Verträge mit Strafklauseln entmutigen notwendige Umfangsänderungen, selbst wenn Änderungen das Produkt verbessern würden.
- [Fehlausgerichtete Liefergegenstände](fehlausgerichtete-liefergegenstaende.md)
<br/>  Vertragsbedingungen, die nicht der technischen Realität entsprechen, produzieren Liefergegenstände, die Vertragsspezifikationen erfüllen, aber tatsächliche Bedürfnisse verfehlen.
- [Rechtsstreitigkeiten](rechtsstreitigkeiten.md)
<br/>  Schlecht designte Verträge schaffen Mehrdeutigkeiten und fehlausgerichtete Erwartungen, die zu Rechtskonflikten eskalieren.
- [Belastete Anbieterbeziehung](belastete-anbieterbeziehung.md)
<br/>  Verträge mit fehlausgerichteten Anreizen oder unrealistischen Bedingungen schaffen Reibung zwischen vertragschließenden Parteien.
- [Qualitätskompromisse](qualitaetskompromisse.md)
<br/>  Wenn Verträge Abweichungen bestrafen, liefern Teams nach Vertragsspezifikation statt nach Qualitätsstandards, was das tatsächliche Produkt kompromittiert.

## Causes ▼

- [Kommunikationslücke zwischen Stakeholdern und Entwicklern](kommunikationsluecke-zwischen-stakeholdern-und-entwicklern.md)
<br/>  Wenn technisches Personal nicht in Vertragsverhandlungen einbezogen wird, spiegeln rechtliche Bedingungen technische Realitäten nicht wider.
- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Verträge, die auf schlecht erhobenen Anforderungen basieren, backen unrealistische Liefergegenstände und Zeitpläne ein.
- [Schlechte Planung](schlechte-planung.md)
<br/>  Unzureichende Projektplanung führt zu Verträgen, die technische Komplexität oder realistische Zeitpläne nicht berücksichtigen.

## Detection Methods ○

- **Vertragsüberprüfungsanalyse:** Bewertung von Vertragsbedingungen gegen bewährte Softwareentwicklungspraktiken
- **Häufigkeit von Änderungsanfragen:** Überwachung, wie oft Vertragsänderungen während Projekten benötigt werden
- **Streitmusteranalyse:** Nachverfolgung wiederkehrender Quellen von Meinungsverschiedenheiten zwischen vertragschließenden Parteien
- **Korrelation des Liefererfolgs:** Vergleich von Projekterfolgsraten mit verschiedenen Vertragsstrukturen
- **Stakeholder-Zufriedenheitsbewertung:** Messung der Zufriedenheit mit Vertragsbedingungen aus technischer und rechtlicher Perspektive

## Examples

Ein Softwareentwicklungsvertrag spezifiziert exakte Bildschirmlayouts und Datenbankschemas als fixe Liefergegenstände, mit Strafklauseln für jede Abweichung. Während der Entwicklung offenbart Nutzertests Usability-Probleme, die Schnittstellenänderungen erfordern, aber die Vertragsstruktur entmutigt notwendige Verbesserungen, weil jede Änderung Neuverhandlung und potenzielle Strafen auslöst. Das Ergebnis ist ein geliefertes System, das Vertragsspezifikationen erfüllt, aber die Nutzerbedürfnisse verfehlt. Ein weiteres Beispiel betrifft einen Wartungsvertrag mit fixen Reaktionszeiten für alle Probleme, unabhängig von Schweregrad oder Komplexität. Dies schafft perverse Anreize, bei denen Anbieter schnelle, aber oberflächliche Korrekturen liefern, um Vertragsbedingungen zu erfüllen, statt Grundursachen von Problemen anzugehen.
