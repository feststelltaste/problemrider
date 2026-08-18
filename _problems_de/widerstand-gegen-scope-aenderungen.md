---
title: Widerstand gegen Scope-Änderungen
description: Notwendige Änderungen am Projekt-Scope werden aufgrund von Prozessbeschränkungen,
  Vertragslimits oder organisatorischer Trägheit vermieden oder abgelehnt.
category:
- Management
- Process
related_problems:
- slug: changing-project-scope
  similarity: 0.75
- slug: resistance-to-change
  similarity: 0.7
- slug: no-formal-change-control-process
  similarity: 0.7
- slug: fear-of-change
  similarity: 0.6
- slug: frequent-changes-to-requirements
  similarity: 0.6
- slug: project-resource-constraints
  similarity: 0.55
solutions:
- change-management-process
- formal-change-control-process
- product-owner
- explicit-prioritization-framework
- definition-of-ready
- regular-stakeholder-demonstrations
- story-mapping
- capacity-based-planning
layout: problem
lang: de
en_slug: scope-change-resistance
---

## Description

Widerstand gegen Scope-Änderungen tritt auf, wenn Organisationen oder Teams notwendige Modifikationen am Projekt-Scope, an Anforderungen oder Liefergegenständen vermeiden, aufgrund prozeduraler Barrieren, Vertragsbeschränkungen oder kultureller Widerstände gegen Veränderung. Dieser Widerstand kann Projekte daran hindern, sich an neue Informationen, sich ändernde Geschäftsbedürfnisse oder entdeckte Anforderungen anzupassen, was potenziell zur Lieferung von Lösungen führt, die die tatsächlichen Bedürfnisse nicht erfüllen.

## Indicators ⟡

- Notwendige Änderungen werden identifiziert, aber aufgrund von Prozessbarrieren nicht implementiert
- Teams fahren mit dem ursprünglichen Scope fort, trotz Belegen, dass er aktuelle Bedürfnisse nicht erfüllen wird
- Änderungsanfragen werden ohne ordentliche Bewertung entmutigt oder abgelehnt
- Vertragsbedingungen verhindern die Anpassung an neue Anforderungen oder Erkenntnisse
- Stakeholder äußern Bedenken zum Scope, aber Änderungen werden nicht verfolgt

## Symptoms ▲

- [Fehlausgerichtete Liefergegenstände](fehlausgerichtete-liefergegenstaende.md)
<br/>  Wenn notwendige Scope-Änderungen abgelehnt werden, entspricht das gelieferte Produkt nicht den sich entwickelnden Stakeholder-Bedürfnissen und tatsächlichen Anforderungen.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Teams implementieren Workarounds, um entdeckte Anforderungen anzugehen, die nicht formal in den Projekt-Scope eingebunden werden können.
- [Unzufriedenheit der Stakeholder](unzufriedenheit-der-stakeholder.md)
<br/>  Stakeholder werden unzufrieden, wenn sie sehen, dass die gelieferte Lösung aktuelle Bedürfnisse nicht erfüllt, weil notwendige Änderungen abgelehnt wurden.
- [Suboptimale Lösungen](suboptimale-loesungen.md)
<br/>  Lösungen, die trotz verändertem Verständnis nach ursprünglichen Spezifikationen gebaut werden, adressieren die tatsächlichen Probleme, die sie lösen sollten, nicht vollständig.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Entwicklungsaufwand wird verschwendet, um Features nach einem veralteten Scope zu bauen, der nicht mehr mit tatsächlichen Anforderungen übereinstimmt.

## Causes ▼

- [Schlechtes Vertragsdesign](schlechtes-vertragsdesign.md)
<br/>  Starre Festpreisverträge machen Scope-Änderungen teuer und prozedural schwierig, was notwendige Anpassungen entmutigt.
- [Freigabe-Abhängigkeiten](freigabe-abhaengigkeiten.md)
<br/>  Umfangreiche Genehmigungsprozesse, die für Scope-Änderungen erforderlich sind, schaffen Barrieren, die Teams davon abhalten, notwendige Modifikationen zu verfolgen.

## Detection Methods ○

- **Analyse von Scope-Änderungsanfragen:** Verfolgung, welche Änderungen angefragt versus welche genehmigt werden
- **Verfolgung der Anforderungsentwicklung:** Überwachung, wie sich Anforderungen über den Projektlebenszyklus ändern
- **Bewertung der Stakeholder-Ausrichtung:** Messung der Ausrichtung zwischen geliefertem Scope und tatsächlichen Bedürfnissen
- **Änderungsgenehmigungsrate:** Berechnung des Prozentsatzes vorgeschlagener Änderungen, die tatsächlich implementiert werden
- **Post-Projekt-Reviews:** Bewertung, ob gelieferte Lösungen sich entwickelnde Geschäftsbedürfnisse erfüllt haben

## Examples

Ein Kundenportal-Entwicklungsprojekt entdeckt durch Nutzertests, dass die ursprünglichen Anforderungen nicht dazu passen, wie Kunden tatsächlich mit dem System interagieren möchten. Der Festpreisvertrag macht jedoch jede Scope-Änderung teuer und zeitaufwendig zur Genehmigung. Statt die Lösung an tatsächliche Nutzerbedürfnisse anzupassen, fährt das Team mit dem ursprünglichen Scope fort und liefert ein System, das Kunden schwer zu nutzen finden. Ein weiteres Beispiel betrifft ein Datenmigrationsprojekt, bei dem neue Datenqualitätsprobleme entdeckt werden, die nicht im ursprünglichen Scope waren. Das Projektteam vermeidet es, diese Probleme anzugehen, weil Änderungsanfragen umfangreiche Dokumentation und Genehmigungsprozesse erfordern, was zu einer Migration führt, die Datenqualitätsanforderungen nicht vollständig erfüllt.
