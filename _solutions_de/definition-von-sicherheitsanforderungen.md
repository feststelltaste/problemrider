---
title: Definition von Sicherheitsanforderungen
description: Erhebung und Dokumentation spezifischer Anforderungen an die
  Informationssicherheit.
category:
- Security
- Requirements
problems:
- inadequate-requirements-gathering
- requirements-ambiguity
- implementation-starts-without-design
- regulatory-compliance-drift
- quality-blind-spots
- frequent-changes-to-requirements
- poor-contract-design
layout: solution
lang: de
en_slug: security-requirements-definition
related_solutions:
- slug: secure-software-development
  similarity: 0.75
- slug: security-certification
  similarity: 0.75
- slug: security-metrics
  similarity: 0.75
- slug: threat-modeling
  similarity: 0.75
- slug: security-relevant-metrics
  similarity: 0.75
- slug: security-architecture-analysis
  similarity: 0.75
---

## Description

Definition von Sicherheitsanforderungen erhebt und dokumentiert spezifische, testbare Sicherheitserwartungen — abgeleitet aus regulatorischen Verpflichtungen, Branchenstandards und organisatorischen Risikobewertungen — als expliziten Teil der Anforderungsmenge statt als implizite Annahme, die niemand aufgeschrieben hat. Legacy-Systeme erreichen häufig eine Compliance-Prüfung oder einen Vorfall nur, um offenzulegen, dass solche Anforderungen nie formal erfasst wurden, was das Team unfähig lässt, mit irgendeiner Zuversicht zu sagen, welche Sicherheitserwartungen das System tatsächlich erfüllt. Anforderungen als testbare Aussagen zu schreiben und sie durch Design, Implementierung und Testing zu verfolgen, verwandelt „wir nehmen an, das ist sicher genug" in eine Gap-Analyse, die priorisiert und bearbeitet werden kann, obwohl die Erhebung von Anforderungen, die umfassend genug sind, um nützlich zu sein, echte Sicherheitsexpertise und enge Zusammenarbeit mit Stakeholdern erfordert, die sich nicht immer über Priorität einig sind.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Leiten Sie Sicherheitsanforderungen aus regulatorischen Verpflichtungen, Branchenstandards und organisatorischen Risikobewertungen ab
- Dokumentieren Sie Sicherheitsanforderungen als testbare Aussagen mit klaren Abnahmekriterien
- Beziehen Sie Sicherheitsanforderungen neben funktionalen Anforderungen in das Produkt-Backlog ein
- Überprüfen Sie Legacy-System-Fähigkeiten gegen dokumentierte Sicherheitsanforderungen, um Lücken zu identifizieren
- Priorisieren Sie Sicherheitsanforderungen basierend auf Risikoauswirkung und Implementierungsmachbarkeit
- Validieren Sie Sicherheitsanforderungen mit Stakeholdern, einschließlich Sicherheits-, Compliance- und Geschäftsteams
- Verfolgen Sie Sicherheitsanforderungen durch Design, Implementierung und Testing, um Abdeckung sicherzustellen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Macht Sicherheitserwartungen explizit und verifizierbar statt impliziter Annahmen
- Verhindert spät angesiedelte Überraschungen, wenn Sicherheitslücken während Audits oder Vorfällen entdeckt werden
- Ermöglicht systematisches Sicherheitstesting gegen definierte Anforderungen
- Schafft Ausrichtung zwischen Sicherheits-, Entwicklungs- und Geschäfts-Stakeholdern

**Kosten und Risiken:**
- Die Erhebung umfassender Sicherheitsanforderungen erfordert Sicherheitsexpertise und Stakeholder-Zusammenarbeit
- Anforderungen können veralten, während sich Bedrohungen und Vorschriften weiterentwickeln
- Überspezifikation kann Implementierungsflexibilität unnötig einschränken
- Legacy-Systeme könnten unfähig sein, bestimmte Sicherheitsanforderungen ohne erhebliche Nacharbeit zu erfüllen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Regierungsbehörde, die ein Legacy-Bürgerdienste-Portal modernisierte, entdeckte während eines Compliance-Reviews, dass nie formale Sicherheitsanforderungen dokumentiert worden waren. Das Team führte eine Reihe von Workshops mit Sicherheitsspezialisten, Rechtsberatern und Systemarchitekten durch, um 45 Sicherheitsanforderungen zu definieren, die Authentifizierung, Datenschutz, Audit-Logging und Zugriffskontrolle abdeckten. Die Abbildung dieser Anforderungen gegen das bestehende System offenbarte, dass 18 vollständig erfüllt, 15 teilweise erfüllt und 12 vollständig unadressiert waren. Diese Gap-Analyse wurde zur Grundlage für eine zweijährige Sicherheitsverbesserungs-Roadmap, die die kritischsten Lücken zuerst priorisierte.
